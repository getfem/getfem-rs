//! gmsh .msh import (ASCII, formats 2.x and 4.1), physical groups as regions.
//!
//! Elements of the highest dimension become convexes; lower-dimensional
//! elements become faces of those convexes in their regions (as GetFEM).
//! Format 4 uses physical tags (as GetFEM); unlike GetFEM, elements in
//! several physical groups are merged. Unlike GetFEM, the mesh is 3D (the
//! format's coordinates) and nodes are not merged (each node used by an
//! element is one point).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::region::RegionSet;

use crate::gmsh_types::ELEMENT_TYPES;
use crate::{Error, Result};

#[derive(Clone, Debug, Default)]
pub struct GmshOptions {
  /// Also add lower-dimensional elements that are not faces as convexes.
  pub add_all_element_types: bool,
  /// Regions whose lower-dimensional elements are added as convexes.
  pub lower_dim_convex_regions: BTreeSet<usize>,
}

#[derive(Debug)]
pub struct GmshMesh {
  pub mesh: Mesh,
  /// Lower-dimensional elements that are not faces of any convex, with their
  /// gmsh type.
  pub ignored: Vec<u32>,
}

struct Elem {
  gmsh_type: u32,
  gt: Option<PGeoTrans>,
  nodes: Vec<usize>,
  regions: Vec<usize>,
}

impl Elem {
  fn dim(&self) -> usize {
    self.gt.as_ref().map_or(0, |g| g.dim())
  }
}

struct Tokens<'a> {
  it: std::iter::Peekable<std::str::SplitAsciiWhitespace<'a>>,
  file: &'a str,
}

impl<'a> Tokens<'a> {
  fn err(&self, m: String) -> Error {
    Error::Format(format!("{}: {m}", self.file))
  }

  fn word(&mut self) -> Result<&'a str> {
    self
      .it
      .next()
      .ok_or_else(|| self.err("unexpected end of file".into()))
  }

  fn num<T: std::str::FromStr>(&mut self) -> Result<T> {
    let w = self.word()?;
    w.parse()
      .map_err(|_| self.err(format!("expected a number, got '{w}'")))
  }

  fn expect(&mut self, s: &str) -> Result<()> {
    let w = self.word()?;
    if w != s {
      return Err(self.err(format!("expected {s}, got '{w}'")));
    }
    Ok(())
  }

  /// Skip to the token after `$End<name>`.
  fn skip_section(&mut self, name: &str) -> Result<()> {
    let end = format!("$End{name}");
    while self.word()? != end {}
    Ok(())
  }
}

fn element_type(t: u32) -> Option<(PGeoTrans, &'static [usize])> {
  ELEMENT_TYPES
    .iter()
    .find(|e| e.0 == t)
    .map(|e| (geotrans_descriptor(e.1).expect("builtin"), e.2))
}

fn nb_nodes(t: u32) -> Option<usize> {
  if t == 15 {
    return Some(1);
  }
  ELEMENT_TYPES.iter().find(|e| e.0 == t).map(|e| e.2.len())
}

pub fn import_from_gmsh(path: &str, opt: &GmshOptions) -> Result<GmshMesh> {
  let text = std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{path}: {e}")))?;
  parse_gmsh(&text, path, opt)
}

/// The points of gmsh nodes, made on first use (no merging).
fn points(
  mesh: &mut MeshData,
  pts: &mut HashMap<usize, usize>,
  coords: &HashMap<usize, [f64; 3]>,
  nodes: &[usize],
) -> Result<Vec<usize>> {
  let mut v = Vec::with_capacity(nodes.len());
  for id in nodes {
    let n = pts.len();
    let i = *pts.entry(*id).or_insert(n);
    if i == n {
      mesh.add_point_at(i, &coords[id])?;
    }
    v.push(i);
  }
  Ok(v)
}

/// Parse gmsh text; `file` is used in error messages.
pub fn parse_gmsh(text: &str, file: &str, opt: &GmshOptions) -> Result<GmshMesh> {
  let mut t = Tokens {
    it: text.split_ascii_whitespace().peekable(),
    file,
  };
  t.expect("$MeshFormat")?;
  let version: f64 = t.num()?;
  let ftype: u32 = t.num()?;
  t.word()?;
  if ftype != 0 {
    return Err(t.err("binary .msh files are not supported, save as ASCII".into()));
  }
  let v4 = if (2.0..3.0).contains(&version) {
    false
  } else if (4.1..5.0).contains(&version) {
    true
  } else {
    return Err(t.err(format!(
      "msh format {version} not supported (2.x and 4.1 are)"
    )));
  };
  t.skip_section("MeshFormat")?;

  let mut mesh = MeshData::new(3);
  // gmsh node id -> coordinates; points made when an element uses them
  let mut node_map: HashMap<usize, [f64; 3]> = HashMap::new();
  let mut entity_phys: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
  let mut elems: Vec<Elem> = vec![];
  let mut seen: HashMap<(u32, Vec<usize>), usize> = HashMap::new();

  let mut push_elem =
    |ty: u32, raw: Vec<usize>, regions: Vec<usize>, elems: &mut Vec<Elem>| -> Result<()> {
      let (gt, nodes) = if ty == 15 {
        (None, raw)
      } else {
        let (g, perm) = element_type(ty)
          .ok_or_else(|| Error::Format(format!("{file}: gmsh element type {ty} not supported")))?;
        (Some(g), perm.iter().map(|&p| raw[p]).collect())
      };
      let mut key = nodes.clone();
      key.sort_unstable();
      match seen.get(&(ty, key.clone())) {
        Some(&i) => elems[i].regions.extend(regions),
        None => {
          seen.insert((ty, key), elems.len());
          elems.push(Elem {
            gmsh_type: ty,
            gt,
            nodes,
            regions,
          });
        }
      }
      Ok(())
    };

  while let Some(sec) = t.it.next() {
    match sec {
      "$PhysicalNames" => {
        let n: usize = t.num()?;
        for _ in 0..n {
          let _dim: usize = t.num()?;
          let _tag: usize = t.num()?;
          let mut name = t.word()?.to_string();
          while !(name.len() > 1 && name.ends_with('"')) {
            name.push(' ');
            name.push_str(t.word()?);
          }
        }
        t.skip_section("PhysicalNames")?;
      }
      "$Entities" if v4 => {
        let counts: Vec<usize> = (0..4).map(|_| t.num()).collect::<Result<_>>()?;
        for (dim, &cnt) in counts.iter().enumerate() {
          for _ in 0..cnt {
            let tag: usize = t.num()?;
            for _ in 0..if dim == 0 { 3 } else { 6 } {
              t.num::<f64>()?;
            }
            let np: usize = t.num()?;
            let phys = (0..np)
              .map(|_| t.num::<i64>().map(|v| v.unsigned_abs() as usize))
              .collect::<Result<Vec<_>>>()?;
            if dim > 0 {
              let nb: usize = t.num()?;
              for _ in 0..nb {
                t.num::<i64>()?;
              }
            }
            entity_phys.insert((dim, tag), phys);
          }
        }
        t.skip_section("Entities")?;
      }
      "$Nodes" => {
        if v4 {
          let (nblocks, _n, _, _): (usize, usize, usize, usize) =
            (t.num()?, t.num()?, t.num()?, t.num()?);
          for _ in 0..nblocks {
            let (dim, _tag, param, n): (usize, usize, usize, usize) =
              (t.num()?, t.num()?, t.num()?, t.num()?);
            let ids: Vec<usize> = (0..n).map(|_| t.num()).collect::<Result<_>>()?;
            for id in ids {
              let x = [t.num()?, t.num()?, t.num()?];
              if param == 1 {
                for _ in 0..dim {
                  t.num::<f64>()?;
                }
              }
              node_map.insert(id, x);
            }
          }
        } else {
          let n: usize = t.num()?;
          for _ in 0..n {
            let id: usize = t.num()?;
            let x = [t.num()?, t.num()?, t.num()?];
            node_map.insert(id, x);
          }
        }
        t.skip_section("Nodes")?;
      }
      "$Elements" => {
        let node = |t: &Tokens, id: usize| {
          node_map
            .contains_key(&id)
            .then_some(id)
            .ok_or_else(|| t.err(format!("unknown node {id}")))
        };
        if v4 {
          let (nblocks, _n, _, _): (usize, usize, usize, usize) =
            (t.num()?, t.num()?, t.num()?, t.num()?);
          for _ in 0..nblocks {
            let (dim, tag, ty, n): (usize, usize, u32, usize) =
              (t.num()?, t.num()?, t.num()?, t.num()?);
            let nn =
              nb_nodes(ty).ok_or_else(|| t.err(format!("gmsh element type {ty} not supported")))?;
            // physical groups as regions; an entity without one: region 0
            // (as msh 2.2), or its tag if the file has no physical group
            let has_phys = entity_phys.values().any(|p| !p.is_empty());
            let regions = match entity_phys.get(&(dim, tag)) {
              Some(p) if !p.is_empty() => p.clone(),
              _ if has_phys => vec![0],
              _ => vec![tag],
            };
            for _ in 0..n {
              t.num::<usize>()?;
              let raw = (0..nn)
                .map(|_| t.num().and_then(|id| node(&t, id)))
                .collect::<Result<Vec<_>>>()?;
              push_elem(ty, raw, regions.clone(), &mut elems)?;
            }
          }
        } else {
          let n: usize = t.num()?;
          for _ in 0..n {
            let (_id, ty, ntags): (usize, u32, usize) = (t.num()?, t.num()?, t.num()?);
            let tags: Vec<usize> = (0..ntags).map(|_| t.num()).collect::<Result<_>>()?;
            let nn =
              nb_nodes(ty).ok_or_else(|| t.err(format!("gmsh element type {ty} not supported")))?;
            let raw = (0..nn)
              .map(|_| t.num().and_then(|id| node(&t, id)))
              .collect::<Result<Vec<_>>>()?;
            push_elem(
              ty,
              raw,
              vec![tags.first().copied().unwrap_or(0)],
              &mut elems,
            )?;
          }
        }
        t.skip_section("Elements")?;
      }
      s if s.starts_with("$") && !s.starts_with("$End") => t.skip_section(&s[1..])?,
      s => return Err(t.err(format!("unexpected '{s}'"))),
    }
  }

  // highest dimension first, then region, then file order (stable)
  let key = |e: &Elem| {
    (
      std::cmp::Reverse(e.dim()),
      e.regions.iter().min().copied().unwrap_or(0),
    )
  };
  elems.sort_by_key(key);
  let top = elems.first().map_or(0, |e| e.dim());
  let mut regions: BTreeMap<usize, RegionSet> = BTreeMap::new();
  let mut ignored = vec![];
  // gmsh node id -> point, made on first use (no merging)
  let mut pts: HashMap<usize, usize> = HashMap::new();
  for e in &elems {
    if let Some(gt) = &e.gt
      && (e.dim() == top
        || e
          .regions
          .iter()
          .any(|r| opt.lower_dim_convex_regions.contains(r)))
    {
      let cv = {
        let p = points(&mut mesh, &mut pts, &node_map, &e.nodes)?;
        mesh.add_convex(gt, &p)?
      };
      for &r in &e.regions {
        regions.entry(r).or_default().add(cv);
      }
      continue;
    }
    let mut found = false;
    let face: Option<Vec<usize>> = e.nodes.iter().map(|id| pts.get(id).copied()).collect();
    for &cv in face
      .as_ref()
      .map_or(&[][..], |f| mesh.convexes_of_point(f[0]))
    {
      let s = mesh.structure_of_convex(cv);
      if s.dim() != e.dim() + 1 {
        continue;
      }
      for f in 0..s.nb_faces() {
        let fp = mesh.ind_points_of_face_of_convex(cv, f);
        if face.as_ref().unwrap().iter().all(|p| fp.contains(p)) {
          for &r in &e.regions {
            regions.entry(r).or_default().add_face(cv, f);
          }
          found = true;
        }
      }
    }
    if !found && let Some(gt) = &e.gt {
      if opt.add_all_element_types {
        let cv = {
          let p = points(&mut mesh, &mut pts, &node_map, &e.nodes)?;
          mesh.add_convex(gt, &p)?
        };
        for &r in &e.regions {
          regions.entry(r).or_default().add(cv);
        }
      } else {
        ignored.push(e.gmsh_type);
      }
    }
  }
  for (id, r) in regions {
    mesh.set_region_set(id, r)?;
  }
  crate::planar_hint(&mesh, file);
  Ok(GmshMesh {
    mesh: mesh.into(),
    ignored,
  })
}
