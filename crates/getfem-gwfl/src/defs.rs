//! Definition store of a workspace, as seen by the GWFL front end: names of
//! variables and data with their shapes, macros, transformations. User
//! functions are global (GetFEM's ga_define_function).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, RwLock};

use crate::parser::MacroDict;

/// GetFEM's predefined neighbor-element transformation (its alias
/// neighbour_elt not supported)
pub(crate) const NEIGHBOR: &str = "neighbor_element";

#[derive(Clone, Debug, PartialEq)]
pub enum Support {
  /// finite element field on a mesh of this dimension
  Fem { mesh_dim: usize },
  /// values at integration points
  ImData,
  /// fixed size (global) value
  Fixed,
}

#[derive(Clone, Debug)]
pub struct VarDef {
  /// data (no test functions) or unknown
  pub is_data: bool,
  pub support: Support,
  /// shape at a point (for Fixed: the shape of the value)
  pub qdims: Vec<usize>,
  pub disabled: bool,
}

impl VarDef {
  pub fn qdim(&self) -> usize {
    self.qdims.iter().product()
  }
}

/// A Rust scalar function of (t, u); u is 0 for functions of one argument.
#[derive(Clone)]
pub struct NativeFn(pub std::sync::Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>);

impl std::fmt::Debug for NativeFn {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "<native>")
  }
}

/// A user scalar function (GetFEM ga_define_function): a GWFL expression of
/// t (and u), a Rust closure, or opaque (neither: GetFEM's C functions).
#[derive(Clone, Debug)]
pub struct UserFn {
  pub nargs: usize,
  pub body: Option<String>,
  pub native: Option<NativeFn>,
  /// derivative w.r.t. t and u: a function name or an expression, "" if none
  pub der1: String,
  pub der2: String,
}

#[derive(Clone, Debug)]
pub struct Defs {
  pub vars: BTreeMap<String, VarDef>,
  pub macros: MacroDict,
  /// interpolate transformations and the (variable, transformation) pairs
  /// they depend on ("": the variable itself)
  pub interpolate_transformations: BTreeMap<String, Vec<(String, String)>>,
  pub elementary_transformations: BTreeSet<String>,
  pub operators: crate::operators::Operators,
}

type Fns = BTreeMap<String, UserFn>;

/// derivatives of built-in functions given as expressions, as functions
static BUILTIN_DERS: LazyLock<Fns> = LazyLock::new(|| {
  let mut d = Fns::new();
  for f in crate::names::FUNCTIONS {
    let n = f.eval.nargs();
    for (k, der) in [(1, f.der1), (2, f.der2)] {
      if k <= n && !der.is_empty() && crate::names::function(der).is_none() {
        register(&mut d, &crate::symbolic::der_name(f.name, k), n, der, 1);
      }
    }
  }
  d
});

/// the user functions of the program (GetFEM: global)
static FUNCTIONS: RwLock<Fns> = RwLock::new(BTreeMap::new());

/// A user function or a function of the derivatives (local: being defined)
fn lookup(local: &Fns, name: &str) -> Option<UserFn> {
  if let Some(u) = local.get(name).or_else(|| BUILTIN_DERS.get(name)) {
    return Some(u.clone());
  }
  FUNCTIONS
    .read()
    .unwrap_or_else(|e| e.into_inner())
    .get(name)
    .cloned()
}

/// A user function (or a function of a built-in's derivative).
pub fn function(name: &str) -> Option<UserFn> {
  lookup(&Fns::new(), name)
}

/// Name of the function giving the k-th partial derivative of f.
fn derivative_name_in(local: &Fns, f: &str, k: usize) -> Option<String> {
  if let Some(b) = crate::names::function(f) {
    let d = if k == 1 { b.der1 } else { b.der2 };
    return match d {
      "" => None,
      d if crate::names::function(d).is_some() => Some(d.to_string()),
      _ => Some(crate::symbolic::der_name(f, k)),
    };
  }
  let u = lookup(local, f)?;
  let d = if k == 1 { u.der1 } else { u.der2 };
  (!d.is_empty()).then_some(d)
}

/// Store a function and, down to `depth` levels, its derivatives.
fn register(out: &mut Fns, name: &str, nargs: usize, body: &str, depth: usize) {
  let mut ders = [String::new(), String::new()];
  if depth > 0
    && let Ok(Some(e)) = crate::parser::parse(body, &MacroDict::default())
  {
    for (k, var) in [(1, "t"), (2, "u")].into_iter().take(nargs) {
      let d = crate::symbolic::diff(&e, var, &|g, j| derivative_name_in(out, g, j));
      if let Ok(d) = d {
        let src = d.map_or_else(|| "0".to_string(), |x| crate::symbolic::to_src(&x));
        let dn = crate::symbolic::der_name(name, k);
        register(out, &dn, nargs, &src, depth - 1);
        ders[k - 1] = dn;
      }
    }
  }
  let [der1, der2] = ders;
  out.insert(
    name.to_string(),
    UserFn {
      nargs,
      body: Some(body.to_string()),
      native: None,
      der1,
      der2,
    },
  );
}

/// Insert the functions of `out` (a function and its derivatives); a name
/// in use: an error.
fn publish(name: &str, out: Fns) -> crate::Result<()> {
  let mut fns = FUNCTIONS.write().unwrap_or_else(|e| e.into_inner());
  if out.keys().any(|k| fns.contains_key(k)) {
    return Err(crate::Error::InvalidArgument(format!(
      "{name}: name already in use"
    )));
  }
  fns.extend(out);
  Ok(())
}

fn check_name(name: &str, nargs: usize) -> crate::Result<()> {
  if !(1..=2).contains(&nargs) {
    return Err(crate::Error::InvalidArgument(format!(
      "{name}: 1 or 2 arguments, not {nargs}"
    )));
  }
  if crate::names::reserved(name).is_some() || BUILTIN_DERS.contains_key(name) {
    return Err(crate::Error::InvalidArgument(format!(
      "{name}: name already in use"
    )));
  }
  Ok(())
}

/// Define f(t) (nargs 1) or f(t, u) (nargs 2) by a GWFL expression, for
/// the whole program (GetFEM ga_define_function); der1, der2: expressions
/// of its partial derivatives (used only if all nargs are given, their own
/// derivatives derived), otherwise derived symbolically. A name in use
/// (also by the same definition): an error.
pub fn define_gwfl_function(
  name: &str,
  nargs: usize,
  body: &str,
  der1: &str,
  der2: &str,
) -> crate::Result<()> {
  check_name(name, nargs)?;
  crate::parser::parse(body, &MacroDict::default())?;
  let mut out = Fns::new();
  let given = [der1, der2];
  if given[..nargs].iter().all(|d| !d.is_empty()) {
    let mut ders = [String::new(), String::new()];
    for (k, d) in given[..nargs].iter().enumerate() {
      crate::parser::parse(d, &MacroDict::default())?;
      ders[k] = crate::symbolic::der_name(name, k + 1);
      register(&mut out, &ders[k], nargs, d, 2);
    }
    let [der1, der2] = ders;
    out.insert(
      name.to_string(),
      UserFn {
        nargs,
        body: Some(body.to_string()),
        native: None,
        der1,
        der2,
      },
    );
  } else {
    register(&mut out, name, nargs, body, 2);
  }
  publish(name, out)
}

/// Define f(t) or f(t, u) by a Rust closure, for the whole program, with
/// closures for its derivatives w.r.t. t and u (fewer than nargs: not
/// differentiable further). Derivatives become functions DER1_name,
/// DER2_name. A name in use: an error.
pub fn define_gwfl_function_native(
  name: &str,
  nargs: usize,
  f: NativeFn,
  derivatives: Vec<NativeFn>,
) -> crate::Result<()> {
  check_name(name, nargs)?;
  if derivatives.len() > nargs {
    return Err(crate::Error::InvalidArgument(format!(
      "{name}: at most as many derivatives as arguments"
    )));
  }
  let mut out = Fns::new();
  let mut ders = [String::new(), String::new()];
  for (k, d) in derivatives.into_iter().enumerate() {
    let dn = crate::symbolic::der_name(name, k + 1);
    let u = UserFn {
      nargs,
      body: None,
      native: Some(d),
      der1: String::new(),
      der2: String::new(),
    };
    out.insert(dn.clone(), u);
    ders[k] = dn;
  }
  let [der1, der2] = ders;
  out.insert(
    name.to_string(),
    UserFn {
      nargs,
      body: None,
      native: Some(f),
      der1,
      der2,
    },
  );
  publish(name, out)
}

/// A function without body nor closure (GetFEM's C functions, in the tests
/// against GetFEM): only its derivative names; kept if the name is in use.
#[doc(hidden)]
pub fn define_opaque_function(name: &str, u: UserFn) {
  let mut fns = FUNCTIONS.write().unwrap_or_else(|e| e.into_inner());
  fns.entry(name.to_string()).or_insert(u);
}

impl Default for Defs {
  fn default() -> Self {
    let mut d = Defs {
      vars: BTreeMap::new(),
      macros: MacroDict::default(),
      interpolate_transformations: BTreeMap::new(),
      elementary_transformations: BTreeSet::new(),
      operators: crate::operators::Operators::default(),
    };
    // predefined transformations of GetFEM's ga_workspace
    d.interpolate_transformations
      .insert(NEIGHBOR.into(), vec![]);
    // predefined macros of GetFEM's ga_workspace
    for (m, e) in [("Hess(u)", "Hess_u"), ("Div(u)", "Div_u")] {
      d.macros.add(m, e).expect("predefined macro");
    }
    d
  }
}

impl Defs {
  /// Name of the function giving the k-th partial derivative of f.
  pub fn derivative_name(&self, f: &str, k: usize) -> Option<String> {
    derivative_name_in(&Fns::new(), f, k)
  }

  pub fn add(&mut self, name: &str, is_data: bool, support: Support, qdims: &[usize]) {
    let v = VarDef {
      is_data,
      support,
      qdims: qdims.to_vec(),
      disabled: false,
    };
    self.vars.insert(name.to_string(), v);
  }

  pub fn var(&self, name: &str) -> Option<&VarDef> {
    self.vars.get(name)
  }
}
