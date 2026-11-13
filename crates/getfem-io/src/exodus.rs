//! Exodus II import and export (docs/design/exodus.md): the 64-bit offset
//! netCDF format, our own reader and writer.
//!
//! Import: the file's dimension; each node used by an element one point
//! (not merged); side sets -> face regions and element sets -> convex
//! regions, numbered by their id (one region if the same id); element
//! blocks -> convex regions 100000 + id; node sets ignored.
//!
//! Export: the fields of one space (one space per file) at the nodes of a
//! continuous Lagrange space of degree <= 2 holding the mesh and that
//! space (an incomplete mesh stays incomplete); blocks: the regions from 100000 if they partition the
//! convexes by element type, else one per element type; the other regions
//! as element sets (convexes) and side sets (faces); nodal fields at time
//! steps, appended to a file of the same mesh, space and fields.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Seek, SeekFrom, Write};

use getfem_fem::fem::{PFem, classical_fem};
use getfem_fem::interpolation::interpolation;
use getfem_fem::mesh_fem::MeshFem;
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::region::RegionSet;

use crate::exodus_types::{ExoType, TYPES, exodus_type, lagrange_type};
use crate::netcdf::{Att, NC_CHAR, NC_DOUBLE, NC_INT, NcFile, NcWriter, doubles, ints, strings};
use crate::vtk::linear_of;
use crate::{Error, Result};

/// The region of an element block: BLOCK_REGIONS + its id.
pub const BLOCK_REGIONS: usize = 100000;

/// An imported element: its convex, its type.
struct Elem {
  cv: usize,
  ty: &'static ExoType,
  /// its nodes' points, Exodus order
  pts: Vec<usize>,
}

/// The ids of an entity kind (`eb`, `ss`, `els`, `ns`): `<kind>_prop1`,
/// else 1, 2, ...
fn ids(nc: &NcFile, kind: &str, n: usize) -> Result<Vec<i64>> {
  match nc.var(&format!("{kind}_prop1")) {
    Some(_) => nc.get_i64(&format!("{kind}_prop1")),
    None => Ok((1..=n as i64).collect()),
  }
}

/// A set id as a region number (< BLOCK_REGIONS).
fn set_region(path: &str, what: &str, id: i64) -> Result<usize> {
  match usize::try_from(id) {
    Ok(r) if r < BLOCK_REGIONS => Ok(r),
    _ => Err(Error::Format(format!(
      "{path}: {what} {id}: ids from 0 to {} only",
      BLOCK_REGIONS - 1
    ))),
  }
}

fn coordinates(nc: &NcFile, dim: usize, n: usize) -> Result<Vec<Vec<f64>>> {
  let c: Vec<Vec<f64>> = if nc.var("coordx").is_some() {
    ["coordx", "coordy", "coordz"][..dim]
      .iter()
      .map(|v| nc.get_f64(v, None))
      .collect::<Result<_>>()?
  } else {
    let all = nc.get_f64("coord", None)?;
    all.chunks(n.max(1)).map(|c| c.to_vec()).collect()
  };
  if c.len() != dim || c.iter().any(|x| x.len() != n) {
    return Err(Error::Format("bad nodal coordinates".into()));
  }
  Ok(c)
}

/// The mesh of an Exodus II file.
pub fn import_from_exodus(path: &str) -> Result<Mesh> {
  let nc = NcFile::open(path)?;
  let err = |m: String| Error::Format(format!("{path}: {m}"));
  let dim = nc
    .dim("num_dim")
    .ok_or_else(|| err("no num_dim: not an Exodus file".into()))?;
  let n = nc.dim("num_nodes").unwrap_or(0);
  let x = coordinates(&nc, dim, n).map_err(|e| err(e.to_string()))?;
  let mut mesh = MeshData::new(dim);
  let mut regions: BTreeMap<usize, RegionSet> = BTreeMap::new();
  // Exodus node -> point, made on first use (no merging)
  let mut pts: HashMap<usize, usize> = HashMap::new();
  let mut elems: Vec<Elem> = vec![];
  let nblk = nc.dim("num_el_blk").unwrap_or(0);
  let blk_ids = ids(&nc, "eb", nblk)?;
  for (b, &id) in blk_ids.iter().enumerate() {
    let conn = format!("connect{}", b + 1);
    if nc.var(&conn).is_none() {
      continue; // an empty block
    }
    let shape = nc.shape(&conn)?;
    let name = nc.att_text(&conn, "elem_type").unwrap_or_default();
    let ty = exodus_type(&name, shape[1])
      .ok_or_else(|| err(format!("block {id}: element type {name} not supported")))?;
    let (gt, perm) = ty.transformation()?;
    let region = BLOCK_REGIONS + usize::try_from(id).map_err(|_| err(format!("block id {id}")))?;
    for e in nc.get_i64(&conn)?.chunks(shape[1]) {
      let mut p = Vec::with_capacity(e.len());
      for &node in e {
        let i = usize::try_from(node - 1)
          .ok()
          .filter(|&i| i < n)
          .ok_or_else(|| err(format!("block {id}: node {node} out of range")))?;
        let k = pts.len();
        let q = *pts.entry(i).or_insert(k);
        if q == k {
          let c: Vec<f64> = x.iter().map(|xs| xs[i]).collect();
          mesh.add_point_at(q, &c)?;
        }
        p.push(q);
      }
      let gp: Vec<usize> = perm.iter().map(|&j| p[j]).collect();
      let cv = mesh.add_convex(&gt, &gp)?;
      regions.entry(region).or_default().add(cv);
      elems.push(Elem { cv, ty, pts: p });
    }
  }
  let elem = |e: i64, what: &str| {
    usize::try_from(e - 1)
      .ok()
      .and_then(|i| elems.get(i))
      .ok_or_else(|| err(format!("{what}: element {e} out of range")))
  };
  // side sets: faces found by their vertices
  let nss = nc.dim("num_side_sets").unwrap_or(0);
  let mut face_regions = BTreeSet::new();
  for (s, &id) in ids(&nc, "ss", nss)?.iter().enumerate() {
    let what = format!("side set {id}");
    let r = set_region(path, &what, id)?;
    face_regions.insert(r);
    let reg = regions.entry(r).or_default();
    let (es, ss) = (format!("elem_ss{}", s + 1), format!("side_ss{}", s + 1));
    if nc.var(&es).is_none() {
      continue;
    }
    for (&e, &side) in nc.get_i64(&es)?.iter().zip(&nc.get_i64(&ss)?) {
      let el = elem(e, &what)?;
      let sides = el.ty.shape.sides;
      // 2D elements in 3D (shells): sides 1, 2 the faces, then the edges
      let shell = el.ty.shape.verts[0].len() < dim && sides.len() > 2;
      let k = usize::try_from(side - 1 - 2 * shell as i64)
        .ok()
        .filter(|&k| k < sides.len())
        .ok_or_else(|| err(format!("{what}: element {e}: side {side} not supported")))?;
      let v: Vec<usize> = sides[k].iter().map(|&i| el.pts[i]).collect();
      let f = (0..mesh.structure_of_convex(el.cv).nb_faces())
        .find(|&f| {
          let fp = mesh.ind_points_of_face_of_convex(el.cv, f);
          v.iter().all(|p| fp.contains(p))
        })
        .ok_or_else(|| err(format!("{what}: element {e}: side {side} not found")))?;
      reg.add_face(el.cv, f);
    }
  }
  // element sets
  let nes = nc.dim("num_elem_sets").unwrap_or(0);
  for (s, &id) in ids(&nc, "els", nes)?.iter().enumerate() {
    let what = format!("element set {id}");
    let r = set_region(path, &what, id)?;
    if face_regions.contains(&r) {
      eprintln!(
        "GetFEM-rs warning: {path}: element set {id} and side set {id} in one \
         region {r} (convexes and faces)"
      );
    }
    let es = format!("elem_els{}", s + 1);
    if nc.var(&es).is_none() {
      continue;
    }
    for &e in &nc.get_i64(&es)? {
      let cv = elem(e, &what)?.cv;
      regions.entry(r).or_default().add(cv);
    }
  }
  let nns = nc.dim("num_node_sets").unwrap_or(0);
  if nns > 0 {
    eprintln!(
      "GetFEM-rs info: {path}: node sets {:?} ignored (no node regions; side \
       sets for boundary conditions)",
      ids(&nc, "ns", nns)?
    );
  }
  if !blk_ids.is_empty() {
    eprintln!(
      "GetFEM-rs warning: {path}: element blocks {blk_ids:?} in the regions \
       {BLOCK_REGIONS} + id"
    );
  }
  for (id, r) in regions {
    mesh.set_region_set(id, r)?;
  }
  crate::planar_hint(&mesh, path);
  Ok(mesh.into())
}

/// Names of the q components of a field (q 1; vectors of 2, 3; 2 x 2 and
/// 3 x 3 tensors, component (i, j) at i + n j; others numbered from 1).
fn component_names(name: &str, q: usize) -> Vec<String> {
  const X: [&str; 3] = ["x", "y", "z"];
  match q {
    1 => vec![name.to_string()],
    2 | 3 => X[..q].iter().map(|c| format!("{name}_{c}")).collect(),
    4 | 9 => {
      let n = if q == 4 { 2 } else { 3 };
      (0..q)
        .map(|k| format!("{name}_{}{}", X[k % n], X[k / n]))
        .collect()
    }
    _ => (1..=q).map(|k| format!("{name}_{k}")).collect(),
  }
}

/// The element blocks (id -> convexes): the regions from BLOCK_REGIONS if
/// they partition the convexes, one element type each (as imported), else
/// one block per element type, numbered from 1.
fn blocks(
  mesh: &MeshData,
  elems: &BTreeMap<usize, (&'static ExoType, Vec<i32>)>,
  path: &str,
) -> BTreeMap<usize, Vec<usize>> {
  let high: Vec<usize> = mesh.region_ids().filter(|&r| r >= BLOCK_REGIONS).collect();
  let mut owner: BTreeMap<usize, usize> = BTreeMap::new();
  let mut ok = !high.is_empty();
  for &r in &high {
    for (cv, f) in mesh.region_set(r).unwrap().iter() {
      ok &= f.is_none() && owner.insert(cv, r - BLOCK_REGIONS).is_none();
    }
  }
  let mut types: BTreeMap<usize, &ExoType> = BTreeMap::new();
  ok &= owner.len() == elems.len()
    && elems
      .iter()
      .all(|(cv, (ty, _))| std::ptr::eq(*types.entry(owner[cv]).or_insert(ty), *ty));
  if !ok && !high.is_empty() {
    eprintln!(
      "GetFEM-rs warning: {path}: regions {high:?} not written (regions from \
       {BLOCK_REGIONS} are blocks: a partition of the convexes, one element \
       type each)"
    );
  }
  let order: Vec<&str> = TYPES
    .iter()
    .map(|t| t.name)
    .filter(|n| elems.values().any(|(ty, _)| ty.name == *n))
    .collect();
  let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
  for (&cv, (ty, _)) in elems {
    let b = if ok {
      owner[&cv]
    } else {
      1 + order.iter().position(|n| *n == ty.name).unwrap()
    };
    blocks.entry(b).or_default().push(cv);
  }
  blocks
}

/// The file's nodes: continuous Lagrange, per convex (of mf, all for a
/// mesh alone) the smallest of degree 1, incomplete degree 2, complete
/// degree 2 holding the transformation and mf's element (degree > 2:
/// complete degree 2, interpolated). An incomplete mesh stays incomplete.
fn exodus_space(path: &str, mesh: &Mesh, mf: Option<&MeshFem>) -> Result<MeshFem> {
  let m = mesh.snapshot();
  let num = mf.map(|mf| mf.numbering()).transpose()?;
  // 1: degree 1, 2: incomplete degree 2, 3: complete degree 2
  let class = |name: &str, linear: bool| match (name.contains("INCOMPLETE"), linear) {
    (true, _) => 2,
    (false, true) => 1,
    _ => 3,
  };
  let cvs: Vec<usize> = match &num {
    Some(n) => n.convex_index().collect(),
    None => m.convex_index().collect(),
  };
  let mut groups: BTreeMap<String, (PFem, Vec<usize>)> = BTreeMap::new();
  // transformations written with other nodes (sub- or superparametric)
  let mut changed: BTreeSet<String> = BTreeSet::new();
  for cv in cvs {
    let pgt = m.trans_of_convex(cv);
    let gt_class = class(&pgt.name(), pgt.complexity() <= 1);
    let p1 = classical_fem(pgt, 1, false)?;
    let pf = num.as_ref().and_then(|n| n.fem_of_element(cv));
    if let Some(pf) = pf
      && pf.dof_types().iter().any(|d| !d.linkable)
    {
      return Err(Error::Format(format!(
        "Exodus export of {}: discontinuous (the file's nodes are shared; \
         export_to_vtu)",
        pf.name()
      )));
    }
    let mf_class = pf.map_or(0, |pf| class(pf.name(), pf.name() == p1.name()));
    let pe = match gt_class.max(mf_class) {
      1 => p1,
      2 if gt_class < 2 => pf.unwrap().clone(),
      2 => classical_fem(pgt, 2, false)?,
      _ => classical_fem(&linear_of(pgt)?, 2, false)?,
    };
    if pe.nb_dof() != pgt.nb_points() {
      let b = pgt.basic_structure();
      let ty = lagrange_type(b.dim(), b.nb_points(), pe.nb_dof()).map_or("?", |t| t.name);
      changed.insert(format!("{} written as {ty}", pgt.name()));
    }
    groups
      .entry(pe.name().to_string())
      .or_insert_with(|| (pe.clone(), vec![]))
      .1
      .push(cv);
  }
  if !changed.is_empty() {
    eprintln!(
      "GetFEM-rs warning: {path}: convexes {} for the space's degree: a \
       reimport gives an isoparametric mesh of these elements, not this mesh",
      changed.into_iter().collect::<Vec<_>>().join(", ")
    );
  }
  let pmf = MeshFem::new(mesh, 1);
  for (pf, cvs) in groups.values() {
    pmf.set_finite_element((cvs.iter().copied(), pf))?;
  }
  Ok(pmf)
}

/// Exodus II file of the fields (u, name) on mf at time t: one space per
/// file (the nodes of exodus_space). append: one more time step at the end
/// of the file, of the same mesh, space and field names.
pub fn export_to_exodus(
  path: &str,
  mf: &MeshFem,
  fields: &[(&[f64], &str)],
  t: f64,
  append: bool,
) -> Result<()> {
  let pmf = exodus_space(path, &mf.linked_mesh(), Some(mf))?;
  let n = pmf.nb_basic_dof()?;
  let mut vals: Vec<(String, Vec<f64>)> = vec![];
  for (u, name) in fields {
    let v = interpolation(mf, &pmf, u)?;
    let q = v.len() / n.max(1);
    for (c, cn) in component_names(name, q).into_iter().enumerate() {
      vals.push((cn, (0..n).map(|d| v[d * q + c]).collect()));
    }
  }
  let w = file(path, &pmf, &vals, Some(t))?;
  if append {
    append_step(path, &w, &vals)
  } else {
    save(path, &w)
  }
}

/// Exodus II file of a mesh alone (its nodes, no time step).
pub fn export_mesh_to_exodus(path: &str, mesh: &Mesh) -> Result<()> {
  save(
    path,
    &file(path, &exodus_space(path, mesh, None)?, &[], None)?,
  )
}

fn save(path: &str, w: &NcWriter) -> Result<()> {
  std::fs::write(path, w.bytes().0).map_err(|e| Error::Io(format!("{path}: {e}")))
}

/// The record of w at the end of the file at path, if the file is w with
/// other records (the header reread: no state kept).
fn append_step(path: &str, w: &NcWriter, vals: &[(String, Vec<f64>)]) -> Result<()> {
  let (bytes, rec) = w.bytes();
  let old = std::fs::read(path).map_err(|e| Error::Io(format!("{path}: {e}")))?;
  if old.len() < rec || old[..4] != bytes[..4] || old[8..rec] != bytes[8..rec] {
    let names: Vec<&str> = vals.iter().map(|f| f.0.as_str()).collect();
    let theirs = NcFile::open(path)
      .and_then(|nc| match nc.var("name_nod_var") {
        Some(_) => nc.get_strings("name_nod_var"),
        None => Ok(vec![]),
      })
      .unwrap_or_default();
    return Err(Error::Format(if theirs != names {
      format!(
        "{path}: fields {} expected (got {})",
        theirs.join(", "),
        names.join(", ")
      )
    } else {
      format!("{path}: another mesh or space")
    }));
  }
  let n = u32::from_be_bytes(old[4..8].try_into().unwrap());
  let off = rec as u64 + n as u64 * (bytes.len() - rec) as u64;
  let mut f = std::fs::OpenOptions::new()
    .write(true)
    .open(path)
    .map_err(|e| Error::Io(format!("{path}: {e}")))?;
  f.set_len(off)
    .and_then(|_| f.seek(SeekFrom::Start(off)))
    .and_then(|_| f.write_all(&bytes[rec..]))
    .and_then(|_| f.seek(SeekFrom::Start(4)))
    .and_then(|_| f.write_all(&(n + 1).to_be_bytes()))
    .map_err(|e| Error::Io(format!("{path}: {e}")))
}

/// The ids of n entities of a kind (`eb`, `ss`, `els`): all active.
fn entities(w: &mut NcWriter, kind: &str, d: usize, ids: &[usize]) {
  let n = ids.len();
  w.var(
    &format!("{kind}_status"),
    &[d],
    NC_INT,
    vec![],
    ints(&vec![1; n]),
  );
  let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
  let a = vec![Att::text("name", "ID")];
  w.var(&format!("{kind}_prop1"), &[d], NC_INT, a, ints(&ids));
}

/// The file of the nodes of pmf (its mesh's regions as sets), the fields
/// at time t (no time step if none).
fn file(
  path: &str,
  pmf: &MeshFem,
  vals: &[(String, Vec<f64>)],
  t: Option<f64>,
) -> Result<NcWriter> {
  let mesh = pmf.linked_mesh().snapshot();
  let dim = mesh.dim();
  if dim > 3 {
    return Err(Error::Format(format!(
      "Exodus export of a mesh of dimension {dim}"
    )));
  }
  let nn = pmf.nb_basic_dof()?;
  let x = (0..nn)
    .map(|d| pmf.point_of_basic_dof(d))
    .collect::<std::result::Result<Vec<_>, _>>()?;
  // the elements: type, connectivity (Exodus order, 1-based)
  let mut elems: BTreeMap<usize, (&ExoType, Vec<i32>)> = BTreeMap::new();
  for cv in pmf.convex_index()? {
    let b = mesh.trans_of_convex(cv).basic_structure();
    let dofs = pmf.ind_basic_dof_of_element(cv)?;
    let ty = lagrange_type(b.dim(), b.nb_points(), dofs.len())
      .ok_or_else(|| Error::Format(format!("Exodus export: no element type for convex {cv}")))?;
    let (_, perm) = ty.transformation()?;
    let mut c = vec![0; dofs.len()];
    for (i, &p) in perm.iter().enumerate() {
      c[p] = dofs[i] as i32 + 1;
    }
    elems.insert(cv, (ty, c));
  }
  let blocks = blocks(&mesh, &elems, path);
  let number: BTreeMap<usize, i32> = blocks
    .values()
    .flatten()
    .enumerate()
    .map(|(k, &cv)| (cv, k as i32 + 1))
    .collect();
  // sets: the regions below BLOCK_REGIONS
  let (mut ssets, mut esets) = (vec![], vec![]);
  for r in mesh.region_ids().filter(|&r| r < BLOCK_REGIONS) {
    let (mut es, mut ss) = (vec![], (vec![], vec![]));
    for (cv, f) in mesh.region_set(r).unwrap().iter() {
      let Some(&e) = number.get(&cv) else { continue };
      let Some(f) = f else {
        es.push(e);
        continue;
      };
      let ty = elems[&cv].0;
      let fv = mesh
        .trans_of_convex(cv)
        .basic_structure()
        .ind_points_of_face(f)
        .to_vec();
      let k = ty
        .side_of_face(&fv)
        .ok_or_else(|| Error::Format(format!("Exodus export: face {f} of convex {cv}")))?;
      let shell = ty.shape.verts[0].len() < dim && ty.shape.sides.len() > 2;
      ss.0.push(e);
      ss.1.push((k + 1 + 2 * shell as usize) as i32);
    }
    if !es.is_empty() {
      esets.push((r, es));
    }
    if !ss.0.is_empty() {
      ssets.push((r, ss));
    }
  }
  let mut w = NcWriter::default();
  w.att(Att::float("api_version", 8.11));
  w.att(Att::float("version", 8.11));
  w.att(Att::int("floating_point_word_size", 8));
  w.att(Att::int("file_size", 1));
  w.att(Att::int("maximum_name_length", 32));
  w.att(Att::int("int64_status", 0));
  w.att(Att::text("title", "GetFEM-rs"));
  let len_name = w.dim("len_name", 33);
  let time = w.dim("time_step", 0);
  let ndim = w.dim("num_dim", dim);
  let nodes = w.dim("num_nodes", nn);
  w.dim("num_elem", elems.len());
  let t: Vec<f64> = t.into_iter().collect();
  w.var("time_whole", &[time], NC_DOUBLE, vec![], doubles(&t));
  let xyz = strings(&["X", "Y", "Z"][..dim], 33);
  w.var("coor_names", &[ndim, len_name], NC_CHAR, vec![], xyz);
  for (k, c) in ["coordx", "coordy", "coordz"][..dim].iter().enumerate() {
    let xs: Vec<f64> = x.iter().map(|p| p[k]).collect();
    w.var(c, &[nodes], NC_DOUBLE, vec![], doubles(&xs));
  }
  let d = w.dim("num_el_blk", blocks.len());
  entities(&mut w, "eb", d, &blocks.keys().copied().collect::<Vec<_>>());
  for (b, cvs) in blocks.values().enumerate() {
    let ty = elems[&cvs[0]].0;
    let ne = w.dim(&format!("num_el_in_blk{}", b + 1), cvs.len());
    let np = w.dim(&format!("num_nod_per_el{}", b + 1), ty.nb_nodes());
    let conn: Vec<i32> = cvs.iter().flat_map(|cv| elems[cv].1.clone()).collect();
    let a = vec![Att::text("elem_type", ty.name)];
    w.var(
      &format!("connect{}", b + 1),
      &[ne, np],
      NC_INT,
      a,
      ints(&conn),
    );
  }
  if !ssets.is_empty() {
    let d = w.dim("num_side_sets", ssets.len());
    entities(
      &mut w,
      "ss",
      d,
      &ssets.iter().map(|s| s.0).collect::<Vec<_>>(),
    );
    for (i, (_, (es, ss))) in ssets.iter().enumerate() {
      let n = w.dim(&format!("num_side_ss{}", i + 1), es.len());
      w.var(&format!("elem_ss{}", i + 1), &[n], NC_INT, vec![], ints(es));
      w.var(&format!("side_ss{}", i + 1), &[n], NC_INT, vec![], ints(ss));
    }
  }
  if !esets.is_empty() {
    let d = w.dim("num_elem_sets", esets.len());
    entities(
      &mut w,
      "els",
      d,
      &esets.iter().map(|s| s.0).collect::<Vec<_>>(),
    );
    for (i, (_, es)) in esets.iter().enumerate() {
      let n = w.dim(&format!("num_ele_els{}", i + 1), es.len());
      w.var(
        &format!("elem_els{}", i + 1),
        &[n],
        NC_INT,
        vec![],
        ints(es),
      );
    }
  }
  if !vals.is_empty() {
    let nv = w.dim("num_nod_var", vals.len());
    let names: Vec<&str> = vals.iter().map(|f| f.0.as_str()).collect();
    w.var(
      "name_nod_var",
      &[nv, len_name],
      NC_CHAR,
      vec![],
      strings(&names, 33),
    );
    for (k, (_, v)) in vals.iter().enumerate() {
      let name = format!("vals_nod_var{}", k + 1);
      w.var(&name, &[time, nodes], NC_DOUBLE, vec![], doubles(v));
    }
  }
  Ok(w)
}

/// The nodal variable `name` of an Exodus file at time step `step` (from
/// 0; negative from the end, -1 the last) on the DOFs of mf (with `_x`, ...
/// for qdim > 1, as exported), and the step's time.
/// No numbering relied on: each file element is the convex with the same
/// vertex coordinates; each DOF of mf on it, at a reference node of the
/// element, takes the value of that file node. mf: Lagrange of the mesh's
/// degree or lower, continuous or not.
pub fn import_point_data(
  path: &str,
  mf: &MeshFem,
  name: &str,
  step: i64,
) -> Result<(Vec<f64>, f64)> {
  let nc = NcFile::open(path)?;
  let err = |m: String| Error::Format(format!("{path}: {m}"));
  let nsteps = nc.dim("time_step").unwrap_or(0);
  let k = if step < 0 { step + nsteps as i64 } else { step };
  let step = usize::try_from(k)
    .ok()
    .filter(|&s| s < nsteps)
    .ok_or_else(|| err(format!("no time step {step} ({nsteps} steps)")))?;
  let time = nc.get_f64("time_whole", Some(step))?[0];
  let num = mf.numbering()?;
  let q = num.qdim();
  let names = match nc.var("name_nod_var") {
    Some(_) => nc.get_strings("name_nod_var")?,
    None => vec![],
  };
  let vals = component_names(name, q)
    .iter()
    .map(|c| match names.iter().position(|n| n == c) {
      Some(k) => nc.get_f64(&format!("vals_nod_var{}", k + 1), Some(step)),
      None => Err(err(format!("no field {c} (fields: {})", names.join(", ")))),
    })
    .collect::<Result<Vec<_>>>()?;
  // the convexes by their vertex coordinates (rounded to 1e-9 of the size)
  let mesh = mf.linked_mesh().snapshot();
  let md = mesh.dim();
  let pts: Vec<usize> = mesh.point_index().collect();
  let size = (0..md)
    .map(|k| {
      let xs = pts.iter().map(|&p| mesh.point(p)[k]);
      xs.clone().fold(f64::MIN, f64::max) - xs.fold(f64::MAX, f64::min)
    })
    .fold(0.0, f64::max)
    .max(f64::MIN_POSITIVE);
  let key = |x: &[f64]| -> Vec<i64> {
    (0..md)
      .map(|k| (x.get(k).copied().unwrap_or(0.0) / (1e-9 * size)).round() as i64)
      .collect()
  };
  let mut convex: HashMap<Vec<Vec<i64>>, usize> = HashMap::new();
  for cv in mesh.convex_index() {
    let gt = mesh.trans_of_convex(cv);
    let ip = mesh.ind_points_of_convex(cv);
    let mut k: Vec<Vec<i64>> = gt
      .vertices()
      .iter()
      .map(|&v| key(mesh.point(ip[v])))
      .collect();
    k.sort();
    convex.insert(k, cv);
  }
  let dim = nc.dim("num_dim").unwrap_or(0);
  let n = nc.dim("num_nodes").unwrap_or(0);
  let x = coordinates(&nc, dim, n).map_err(|e| err(e.to_string()))?;
  let node_key = |i: usize| key(&x.iter().map(|xs| xs[i]).collect::<Vec<_>>());
  let mut basic = vec![f64::NAN; num.nb_basic_dof()];
  let mut e = 0;
  for b in 0..nc.dim("num_el_blk").unwrap_or(0) {
    let conn = format!("connect{}", b + 1);
    if nc.var(&conn).is_none() {
      continue;
    }
    let shape = nc.shape(&conn)?;
    let tname = nc.att_text(&conn, "elem_type").unwrap_or_default();
    let ty = exodus_type(&tname, shape[1])
      .ok_or_else(|| err(format!("element type {tname} not supported")))?;
    let nv = ty.shape.verts.len();
    for c in nc.get_i64(&conn)?.chunks(shape[1]) {
      e += 1;
      let nodes: Vec<usize> = c.iter().map(|&i| (i - 1) as usize).collect();
      if nodes.iter().any(|&i| i >= n) {
        return Err(err(format!("element {e}: node out of range")));
      }
      let vk: Vec<Vec<i64>> = nodes[..nv].iter().map(|&i| node_key(i)).collect();
      let mut sorted = vk.clone();
      sorted.sort();
      let cv = *convex
        .get(&sorted)
        .ok_or_else(|| err(format!("element {e}: no convex with its vertices")))?;
      // the reference positions of the element's nodes in the convex
      let gt = mesh.trans_of_convex(cv);
      let ip = mesh.ind_points_of_convex(cv);
      let refs: Vec<Vec<f64>> = vk
        .iter()
        .map(|k| {
          let v = gt
            .vertices()
            .iter()
            .find(|&&v| key(mesh.point(ip[v])) == *k)
            .unwrap();
          gt.geometric_nodes()[*v].clone()
        })
        .collect();
      let centre = |s: &[usize]| -> Vec<f64> {
        (0..refs[0].len())
          .map(|k| s.iter().map(|&i| refs[i][k]).sum::<f64>() / s.len() as f64)
          .collect()
      };
      let refs: Vec<Vec<f64>> = refs
        .iter()
        .cloned()
        .chain(ty.extra.iter().map(|s| centre(s)))
        .collect();
      let Some(pf) = num.fem_of_element(cv) else {
        continue;
      };
      if !pf.is_lagrange() || pf.target_dim() != 1 {
        return Err(err(format!("{}: not a Lagrange element", pf.name())));
      }
      let dofs = num.ind_basic_dof_of_element(cv);
      for i in 0..pf.nb_dof() {
        let r = pf.node(i);
        let j = refs
          .iter()
          .position(|x| x.iter().zip(r).all(|(a, b)| (a - b).abs() < 1e-10))
          .ok_or_else(|| {
            err(format!(
              "{}: a DOF not at a node of element {e} ({}; Lagrange of its degree or lower)",
              pf.name(),
              ty.name
            ))
          })?;
        for (c, v) in vals.iter().enumerate() {
          basic[dofs[i * q + c]] = v[nodes[j]];
        }
      }
    }
  }
  if let Some(d) = basic.iter().position(|v| v.is_nan()) {
    return Err(err(format!(
      "basic DOF {d} of the space: not on an element of the file"
    )));
  }
  Ok((mf.reduce_vector(&basic)?, time))
}
