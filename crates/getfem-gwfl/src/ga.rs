//! GetFEM's compiled instructions (getfem_generic_assembly_compile_and_exec
//! .cc): an analysed tree becomes an instruction set of small instructions
//! on tensors of one buffer, in two phases (first point of an element, each
//! point), executed point by point. Instruction sets using what is not
//! ported yet are refused (None).

use std::ops::Range;
use std::sync::Arc;

use getfem_fem::precomp::GeoPrecomp;
use getfem_geom::geotrans::{PGeoTrans, compute_K_matrix, pseudo_inverse_small};
use getfem_linalg::tensor::Mat;
use getfem_xsmm::{Gemm, GemmShape};

use crate::defs::{Defs, Support};
use crate::lexer::Tok;
use crate::operators::NonlinearOperator;
use crate::semantic::{Der, Src, Tree, Ty};
use crate::{Error, Result};

mod fuse;
pub(crate) mod rewrite;
#[cfg(test)]
pub(crate) use fuse::{NO_FUSE, TEST_GEMM_MIN};

/// index of a tensor of an instruction set
pub(crate) type TensorId = u32;

/// no tensor
pub(crate) const NONE: TensorId = u32::MAX;

/// An instruction: an opcode and a few tensor indices (plain, small); t the
/// result, a, b the arguments (GetFEM's tc1, tc2), v a variable.
#[derive(Clone, Copy, Debug, PartialEq)]
#[rustfmt::skip]
pub(crate) enum Instruction {
  /// the scalar assembly: acc += coeff t[a]
  AddToCoeff { a: u32 },
  /// point coordinates, or one of them
  X { t: u32 },
  XComp { t: u32, k: u32 },
  /// t = a + b, a - b
  Add { t: u32, a: u32, b: u32 },
  Sub { t: u32, a: u32, b: u32 },
  /// t = c a, a / c (c a scalar tensor)
  ScalarMult { t: u32, a: u32, c: u32 },
  ScalarDiv { t: u32, a: u32, c: u32 },
  /// t[i + s1 j] = a[i] b[j]
  SimpleTmult { t: u32, a: u32, b: u32 },
  /// t[m + M n] = sum_i a[n + N i] b[m + M i]
  Contraction { t: u32, a: u32, b: u32, i: u32 },
  /// t[m + M k] = sum_j a[m + M j] b[j + J k]
  MatrixMult { t: u32, a: u32, b: u32, j: u32 },
  /// t[i] = f(a[i]), f in the table of functions
  Func1 { t: u32, a: u32, f: u32 },
  /// a nonlinear operator (value or derivative) of a (and b)
  Nonlin { t: u32, a: u32, b: u32, op: u32 },
  /// t = a (GetFEM's copy_tensor)
  Copy { t: u32, a: u32 },
  /// t += a (the blocks of the condensation)
  AddTo { t: u32, a: u32 },
  /// point: condensation of cluster c (GetFEM
  /// ga_instruction_condensation_sub): K'_QJ = K_QQ^-1 K_QJ, R'_Q =
  /// K_QQ^-1 R_Q / weight, written to the outputs
  Condense { c: u32 },
  /// t = -K_IQ K'_QJ of correction k (GetFEM condensation_super_K)
  CondenseK { t: u32, k: u32 },
  /// t = -K_IQ R'_Q of correction k (GetFEM condensation_super_R)
  CondenseR { t: u32, k: u32 },
  /// point: t times the weight on the condensed right-hand side
  CondVecAssemblyPt { t: u32, v: u32 },
  /// trace over the last two proper indices, of size n
  Trace { t: u32, a: u32, n: u32 },
  /// divergence of variable v (qdim = N), of its test functions
  Diverg { t: u32, z: u32, c: u32, v: u32 },
  CopyDivergBase { t: u32, z: u32, v: u32 },
  /// t[r L + l] = a[map[r] L + l]: components of a (tensor slice), L the
  /// test functions
  Gather { t: u32, a: u32, map: u32 },
  /// t[i] = f(a[i], b[i]) (a or b broadcast if scalar)
  Func2 { t: u32, a: u32, b: u32, f: u32 },
  /// a scalar function given by an expression: expression function f on each
  /// entry of a (and b)
  FuncExpr { t: u32, a: u32, b: u32, f: u32 },
  /// t = f(a, b), f a native function (b: NONE for one argument)
  FuncNative { t: u32, a: u32, b: u32, f: u32 },
  /// explicit tensor: component k from the tensor maps[l][k] (scalars, or
  /// with test functions: broadcast if scalar)
  CMatrix { t: u32, l: u32 },
  CMatrixTests { t: u32, l: u32 },
  /// GetFEM's dotmult, dotdiv (b without test functions), dotmult_spec
  DotMult { t: u32, a: u32, b: u32 },
  DotDiv { t: u32, a: u32, b: u32 },
  DotMultSpec { t: u32, a: u32, b: u32 },
  /// sym, skew of the last two proper indices (s = s1 | s2 << 16),
  /// deviator of size n
  Sym { t: u32, a: u32, s: u32 },
  Skew { t: u32, a: u32, s: u32 },
  Deviator { t: u32, a: u32, n: u32 },
  /// index operations, sizes in maps[l] (GetFEM's swap_indices [nn1,
  /// nn2, ii2, ii3], index_move_last [nn, ii2], contract_1_1 [nn, ii2,
  /// ii3], contract_2_1 [nn, ii1, ii2, ii3, ii4])
  SwapInd { t: u32, a: u32, l: u32 },
  IndMoveLast { t: u32, a: u32, l: u32 },
  Contract11 { t: u32, a: u32, l: u32 },
  Contract21 { t: u32, a: u32, b: u32, l: u32 },
  /// cross product, with test functions (inv: test2 first)
  Cross { t: u32, a: u32, b: u32 },
  CrossTf { t: u32, a: u32, b: u32, inv: u32 },
  /// element_size, element_K, element_B
  EltSize { t: u32 },
  /// point: Hessians of the basis of v, Hessian of v, of its tests
  HessBase { v: u32, z: u32 },
  Hess { t: u32, z: u32, c: u32, v: u32 },
  CopyHessBase { t: u32, z: u32, v: u32 },
  /// point: values of im_data v
  ImVal { t: u32, v: u32 },
  /// point: transformation tr at the point (y: the target point computed,
  /// NONE: neighbor element)
  TransCall { tr: u32, y: u32 },
  InterpX { t: u32, tr: u32 },
  /// point: derivative of the target of contact transformation tr, tests
  /// of v
  TransDer { t: u32, tr: u32, v: u32 },
  InterpNormal { t: u32, tr: u32 },
  /// point: K (b = 0) or B (b = 1) of the target element
  InterpElt { t: u32, tr: u32, b: u32 },
  /// point: variable v (or its tests) on the target element, der: Val, Grad,
  /// Hess, Div
  InterpField { t: u32, v: u32, der: u8, test: u8 },
  /// point: the next len instructions where the status of tr is st (NONE:
  /// not 0), else t = 0 (GetFEM ga_instruction_interpolate_filter)
  Filter { t: u32, tr: u32, st: u32, len: u32 },
  /// element: matrix m of elementary transformation e; coefficients
  /// t = m c; point: tests t = m^T a (GetFEM ga_instruction_elementary_trans)
  EltTrans { m: u32, e: u32 },
  ElemCoef { t: u32, m: u32, c: u32 },
  ElemTest { t: u32, a: u32, m: u32, d: u32 },
  /// point: the values of a (GetFEM ga_instruction_interpolate)
  PointValue { a: u32 },
  /// assembly at each point (tests on target elements)
  VecAssemblyPt { t: u32, v: u32 },
  MatAssemblyPt { t: u32, v1: u32, v2: u32 },
  EltK { t: u32 },
  EltB { t: u32 },
  /// element: the local coefficients of variable v into c
  SliceDofs { v: u32, c: u32 },
  /// point: values, gradients of the basis of variable v
  ValBase { v: u32, z: u32 },
  GradBase { v: u32, z: u32 },
  /// point: values (der 0), gradients (1), Hessians (2) of the basis of
  /// variable v on the side of the level set the point is on (GetFEM
  /// ga_instruction_xfem_{plus,minus}_{val,grad,hess}_base)
  XfemBase { v: u32, z: u32, der: u8, side: i8 },
  /// point: value, gradient of variable v from its basis and coefficients
  Val { t: u32, z: u32, c: u32, v: u32 },
  Grad { t: u32, z: u32, c: u32, v: u32 },
  /// point: the test functions of variable v (value, gradient) from its
  /// basis: t(i Q + j, j [, l]) = Z(i [, l]), Q its qdim
  CopyValBase { t: u32, z: u32, v: u32 },
  CopyGradBase { t: u32, z: u32, v: u32 },
  /// unit outward normal
  Normal { t: u32 },
  /// the first two proper indices swapped, test indices first (sizes of
  /// the two: jk = j | k << 16, i the product of the other proper ones)
  Transpose { t: u32, a: u32, jk: u32, i: u32 },
  /// vector assembly of t (test functions of v), element vector e
  VecAssembly { t: u32, v: u32, e: u32 },
  /// GetFEM's spec_contraction, spec2_contraction (a, b with test
  /// functions, nn the contracted size)
  SpecContraction { t: u32, a: u32, b: u32, nn: u32 },
  Spec2Contraction { t: u32, a: u32, b: u32, nn: u32 },
  /// matrix_mult_spec, matrix_mult_spec2 (jik = J | I << 10 | K << 20)
  MatrixMultSpec { t: u32, a: u32, b: u32, jik: u32 },
  MatrixMultSpec2 { t: u32, a: u32, b: u32, jik: u32 },
  /// spec_tmult (ij = I | J << 16), spec2_tmult
  SpecTmult { t: u32, a: u32, b: u32, ij: u32 },
  Spec2Tmult { t: u32, a: u32, b: u32 },
  /// matrix assembly of t (test functions of v1, v2), element matrix e
  MatAssembly { t: u32, v1: u32, v2: u32, e: u32 },
  /// vectorized test functions (qdim of v > 1): only the nonzeros written
  /// (the tensor is zero from its layout); sparsity 1, 2
  CopyVectValBase { t: u32, z: u32, v: u32 },
  CopyVectGradBase { t: u32, z: u32, v: u32 },
  /// GetFEM's contraction_opt1_1 (both of sparsity 1, result of sparsity
  /// 10), opt0_1 (b of sparsity 1), opt0_2, opt2_0 (b, a of sparsity 2;
  /// nq = n | q << 16)
  ContractionOpt11 { t: u32, a: u32, b: u32, nn: u32 },
  ContractionOpt01 { t: u32, a: u32, b: u32, nn: u32 },
  ContractionOpt02 { t: u32, a: u32, b: u32, nq: u32 },
  ContractionOpt20 { t: u32, a: u32, b: u32, nq: u32 },
  /// GetFEM's contraction_opt2_2: a, b of sparsity 2 fully contracted, the
  /// result of sparsity 10 (one scalar block, on each component)
  ContractionOpt22 { t: u32, a: u32, b: u32, nq: u32 },
  /// matrix assembly of a t of sparsity 10: one scalar block, scattered
  /// on each component (matrix_assembly_standard_vector_opt10)
  MatAssemblyOpt10 { t: u32, v1: u32, v2: u32, e: u32 },
  /// matrix assembly fused over the points of the element (fuse.rs)
  FusedAssembly { f: u32 },
}

impl Instruction {
  /// the tensor written (none for the assembly instructions)
  pub fn out(&self) -> Option<TensorId> {
    match *self {
      Instruction::X { t, .. }
      | Instruction::XComp { t, .. }
      | Instruction::Add { t, .. }
      | Instruction::Sub { t, .. }
      | Instruction::ScalarMult { t, .. }
      | Instruction::ScalarDiv { t, .. }
      | Instruction::SimpleTmult { t, .. }
      | Instruction::Contraction { t, .. }
      | Instruction::MatrixMult { t, .. }
      | Instruction::Func1 { t, .. }
      | Instruction::Nonlin { t, .. }
      | Instruction::Copy { t, .. }
      | Instruction::AddTo { t, .. }
      | Instruction::CondenseK { t, .. }
      | Instruction::CondenseR { t, .. }
      | Instruction::Trace { t, .. }
      | Instruction::Diverg { t, .. }
      | Instruction::CopyDivergBase { t, .. }
      | Instruction::Gather { t, .. }
      | Instruction::Func2 { t, .. }
      | Instruction::FuncExpr { t, .. }
      | Instruction::FuncNative { t, .. }
      | Instruction::CMatrix { t, .. }
      | Instruction::CMatrixTests { t, .. }
      | Instruction::DotMult { t, .. }
      | Instruction::DotDiv { t, .. }
      | Instruction::DotMultSpec { t, .. }
      | Instruction::Sym { t, .. }
      | Instruction::Skew { t, .. }
      | Instruction::Deviator { t, .. }
      | Instruction::SwapInd { t, .. }
      | Instruction::IndMoveLast { t, .. }
      | Instruction::Contract11 { t, .. }
      | Instruction::Contract21 { t, .. }
      | Instruction::Cross { t, .. }
      | Instruction::CrossTf { t, .. }
      | Instruction::EltSize { t, .. }
      | Instruction::Hess { t, .. }
      | Instruction::CopyHessBase { t, .. }
      | Instruction::ImVal { t, .. }
      | Instruction::InterpX { t, .. }
      | Instruction::TransDer { t, .. }
      | Instruction::InterpNormal { t, .. }
      | Instruction::InterpElt { t, .. }
      | Instruction::InterpField { t, .. }
      | Instruction::Filter { t, .. }
      | Instruction::ElemCoef { t, .. }
      | Instruction::ElemTest { t, .. }
      | Instruction::EltK { t, .. }
      | Instruction::EltB { t, .. }
      | Instruction::Val { t, .. }
      | Instruction::Grad { t, .. }
      | Instruction::CopyValBase { t, .. }
      | Instruction::CopyGradBase { t, .. }
      | Instruction::Normal { t, .. }
      | Instruction::Transpose { t, .. }
      | Instruction::SpecContraction { t, .. }
      | Instruction::Spec2Contraction { t, .. }
      | Instruction::MatrixMultSpec { t, .. }
      | Instruction::MatrixMultSpec2 { t, .. }
      | Instruction::SpecTmult { t, .. }
      | Instruction::Spec2Tmult { t, .. }
      | Instruction::CopyVectValBase { t, .. }
      | Instruction::CopyVectGradBase { t, .. }
      | Instruction::ContractionOpt11 { t, .. }
      | Instruction::ContractionOpt01 { t, .. }
      | Instruction::ContractionOpt02 { t, .. }
      | Instruction::ContractionOpt20 { t, .. }
      | Instruction::ContractionOpt22 { t, .. } => Some(t),
      Instruction::HessBase { z, .. }
      | Instruction::ValBase { z, .. }
      | Instruction::GradBase { z, .. }
      | Instruction::XfemBase { z, .. } => Some(z),
      Instruction::EltTrans { m, .. } => Some(m),
      Instruction::SliceDofs { c, .. } => Some(c),
      _ => None,
    }
  }

  /// the tensors read (a superset: NONE and fields that are not tensors
  /// may appear; the components of CMatrix are in InstructionSet::reads)
  fn tensor_args(&self) -> Vec<TensorId> {
    match *self {
      Instruction::AddToCoeff { a, .. }
      | Instruction::Func1 { a, .. }
      | Instruction::Copy { a, .. }
      | Instruction::Trace { a, .. }
      | Instruction::Gather { a, .. }
      | Instruction::Sym { a, .. }
      | Instruction::Skew { a, .. }
      | Instruction::Deviator { a, .. }
      | Instruction::SwapInd { a, .. }
      | Instruction::IndMoveLast { a, .. }
      | Instruction::Contract11 { a, .. }
      | Instruction::PointValue { a, .. }
      | Instruction::Transpose { a, .. } => vec![a],
      Instruction::Add { a, b, .. }
      | Instruction::Sub { a, b, .. }
      | Instruction::SimpleTmult { a, b, .. }
      | Instruction::Contraction { a, b, .. }
      | Instruction::MatrixMult { a, b, .. }
      | Instruction::Nonlin { a, b, .. }
      | Instruction::Func2 { a, b, .. }
      | Instruction::FuncExpr { a, b, .. }
      | Instruction::FuncNative { a, b, .. }
      | Instruction::DotMult { a, b, .. }
      | Instruction::DotDiv { a, b, .. }
      | Instruction::DotMultSpec { a, b, .. }
      | Instruction::Contract21 { a, b, .. }
      | Instruction::Cross { a, b, .. }
      | Instruction::CrossTf { a, b, .. }
      | Instruction::SpecContraction { a, b, .. }
      | Instruction::Spec2Contraction { a, b, .. }
      | Instruction::MatrixMultSpec { a, b, .. }
      | Instruction::MatrixMultSpec2 { a, b, .. }
      | Instruction::SpecTmult { a, b, .. }
      | Instruction::Spec2Tmult { a, b, .. }
      | Instruction::ContractionOpt11 { a, b, .. }
      | Instruction::ContractionOpt01 { a, b, .. }
      | Instruction::ContractionOpt02 { a, b, .. }
      | Instruction::ContractionOpt20 { a, b, .. }
      | Instruction::ContractionOpt22 { a, b, .. } => vec![a, b],
      Instruction::ScalarMult { a, c, .. } | Instruction::ScalarDiv { a, c, .. } => vec![a, c],
      Instruction::AddTo { t, a } => vec![t, a],
      Instruction::CondVecAssemblyPt { t, .. } => vec![t],
      Instruction::Diverg { z, c, .. }
      | Instruction::Hess { z, c, .. }
      | Instruction::Val { z, c, .. }
      | Instruction::Grad { z, c, .. } => vec![z, c],
      Instruction::CopyDivergBase { z, .. }
      | Instruction::CopyHessBase { z, .. }
      | Instruction::CopyValBase { z, .. }
      | Instruction::CopyGradBase { z, .. }
      | Instruction::CopyVectValBase { z, .. }
      | Instruction::CopyVectGradBase { z, .. } => vec![z],
      Instruction::TransCall { y, .. } => vec![y],
      Instruction::ElemCoef { m, c, .. } => vec![m, c],
      Instruction::ElemTest { a, m, .. } => vec![a, m],
      Instruction::VecAssemblyPt { t, .. }
      | Instruction::MatAssemblyPt { t, .. }
      | Instruction::VecAssembly { t, .. }
      | Instruction::MatAssembly { t, .. }
      | Instruction::MatAssemblyOpt10 { t, .. } => vec![t],
      _ => vec![],
    }
  }
}

/// Size of a tensor: fixed (initial values at init in the constants), or
/// from the element of a variable
#[derive(Clone, Copy, Debug)]
pub(crate) enum Shape {
  Fixed {
    len: u32,
    init: u32,
  },
  /// proper size times the test dimensions of variables t1, t2 (NONE: no
  /// test)
  Test {
    proper: u32,
    t1: u32,
    t2: u32,
  },
  BaseVal(u32),
  BaseGrad(u32),
  BaseHess(u32),
  Coef(u32),
  /// matrix of an elementary transformation: tests of dest x tests of src
  ElemMat(u32, u32),
}

/// A cluster of internal variables condensed together at each point: the
/// variables q, the primary variables j coupled to them; the blocks K(q_a, q_b) at a + nq b,
/// K(q_a, j_b) and their outputs K'(q_a, j_b) at a + nq b, the outputs
/// R'(q_a) (NONE: no block)
#[derive(Clone, Debug, Default)]
pub(crate) struct Cond {
  q: Vec<u32>,
  j: Vec<u32>,
  kqq: Vec<u32>,
  kqj: Vec<u32>,
  kqjp: Vec<u32>,
  rqp: Vec<u32>,
}

/// A correction of the condensed system from cluster c: the blocks
/// K(i, q_a) (NONE: none); jb: index of j in the cluster (NONE: the
/// right-hand side)
#[derive(Clone, Debug)]
pub(crate) struct CondK {
  c: u32,
  jb: u32,
  kiq: Vec<u32>,
}

/// A nonlinear operator call.
#[derive(Clone)]
pub(crate) struct NlCall {
  pub op: Arc<dyn NonlinearOperator>,
  pub der: (usize, usize),
  pub shapes: Vec<Vec<usize>>,
}

#[derive(Clone, Default)]
pub(crate) struct InstructionSet {
  pub elt_instructions: Vec<Instruction>,
  pub instructions: Vec<Instruction>,
  pub shapes: Vec<Shape>,
  pub consts: Vec<f64>,
  pub funcs: Vec<fn(f64) -> f64>,
  pub nonlins: Vec<NlCall>,
  pub funcs2: Vec<fn(f64, f64) -> f64>,
  /// Rust closures of (t, u) (user native functions)
  pub natives: Vec<crate::defs::NativeFn>,
  /// index maps of the tensor slices
  pub maps: Vec<Vec<u32>>,
  /// the expression functions, the input tensors of each
  pub subs: Vec<GaFunction>,
  pub inputs: Vec<u32>,
  /// finite element variables read
  pub vars: Vec<String>,
  /// per tensor: GetFEM's sparsity (1, 2: vectorized test value, gradient;
  /// 10: the product of two of sparsity 1), qdim, and for sparsity 2 the
  /// size of the gradient index (compilation only)
  sparsity: Vec<(u8, u8, u8)>,
  /// compiled subtrees by structure: an equal one shares the tensor
  /// (GetFEM: node_list, sub_tree_are_equal, set_to_copy)
  node_list: std::collections::HashMap<u32, u32>,
  /// the copied test functions by (variable, derivative): the other test
  /// function of the same variable shares the copy (GetFEM:
  /// sub_tree_are_equal version 2, set_to_copy)
  tests_copied: std::collections::HashMap<(u32, Der, i8), u32>,
  /// per tensor: the tensor whose storage it shares (its own shape, the same
  /// values)
  pub alias: std::collections::HashMap<u32, u32>,
  /// subtree signatures and their ids; ids of the trees being compiled
  sigs: std::collections::HashMap<Sig, u32>,
  trees: Vec<(usize, Vec<u32>)>,
  /// the points' coordinates, the normal are read
  pub needs_x: bool,
  pub needs_normal: bool,
  /// element_size, K and B are read
  pub needs_elt_size: bool,
  pub needs_kb: bool,
  /// Hessians of basis functions read
  pub needs_hess: bool,
  /// values at the points, no assembly
  pub values: bool,
  /// a scalar assembly (AddToCoeff)
  pub scalar: bool,
  /// elementary transformations (name, source, dest) and their matrices
  pub elts: Vec<(String, u32, u32)>,
  elt_mats: Vec<u32>,
  /// interpolate transformations; strict: used outside filters
  pub trans: Vec<String>,
  strict: Vec<bool>,
  /// contact transformations; derivatives of the target read
  contact: Vec<bool>,
  trans_ders: Vec<bool>,
  in_filter: u32,
  /// element vectors of the assembly instructions
  pub nb_elems: usize,
  /// clusters of internal variables and the corrections of the condensed
  /// system (GetFEM condensation_description)
  conds: Vec<Cond>,
  cond_k: Vec<CondK>,
  /// assemblies fused over the points (FusedAssembly)
  fused: Vec<fuse::Fused>,
  /// per variable: tensors of the coefficients and of the base values
  coef: Vec<u32>,
  base_val: Vec<u32>,
  base_grad: Vec<u32>,
  base_hess: Vec<u32>,
  /// the bases on a side of a level set by (variable, derivative, side)
  xfem_base: std::collections::HashMap<(u32, Der, i8), u32>,
  /// per variable: the sides its real values are needed on (bits 1: +,
  /// 2: -)
  pub xfem_sides: Vec<u8>,
}

/// The element of a variable: basis functions, target dimension, qdim of
/// the variable, dimension of the mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct VarDims {
  pub nb: usize,
  pub td: usize,
  pub qdim: usize,
  pub n: usize,
}

impl VarDims {
  /// test functions: basis functions times qdim / target_dim
  pub fn tdim(&self) -> usize {
    self.nb * (self.qdim / self.td.max(1))
  }
}

/// A tensor range of the buffer: offset and length.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TensorRange {
  pub off: u32,
  pub len: u32,
}

/// Execution state: the buffer and its tensors (laid out for the elements of
/// the variables, dims), the point weight and coordinates, the scalar
/// result.
#[derive(Default)]
pub(crate) struct ExecState {
  /// the tensors of the instruction set, one after the other
  pub buffer: Vec<f64>,
  pub ranges: Vec<TensorRange>,
  pub dims: Vec<VarDims>,
  pub coeff: f64,
  /// point, unit normal (sized n by the assembly)
  pub x: Vec<f64>,
  pub normal: Vec<f64>,
  /// element_size, K, B (n x p, column-major)
  pub elt_size: f64,
  pub k: Vec<f64>,
  pub b: Vec<f64>,
  /// point index in the element, number of points
  pub ipt: usize,
  pub nbpt: usize,
  pub acc: f64,
  pub elems: Vec<Vec<f64>>,
  /// element tensor at a point (per point assembly)
  pub pt_elem: Vec<f64>,
  /// operands of the fused assemblies (per term), their element matrix
  pub stash: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
  pub facc: Vec<Vec<f64>>,
  /// terms by GEMM, element matrix size, for (points, layout)
  pub fgemm: Vec<Vec<bool>>,
  pub flen: Vec<usize>,
  pub fkey: Vec<(usize, u32)>,
  /// layouts done (a new one changes the sizes)
  pub layouts: u32,
  /// condensation: work (K_QQ^-1, R_Q, a column block)
  pub cw: Vec<f64>,
  /// execution states of the expression functions
  pub subs: Vec<ExecState>,
}

/// What the instructions read from the workspace: local coefficients and
/// base values (at the current element and point).
pub(crate) trait Ext {
  fn slice_dofs(&mut self, v: usize, out: &mut [f64]);
  fn val_base(&mut self, v: usize, out: &mut [f64]);
  fn grad_base(&mut self, v: usize, out: &mut [f64]);
  fn hess_base(&mut self, v: usize, out: &mut [f64]) -> Result<()>;
  /// values (der 0), gradients (1), Hessians (2) of the basis of v on a
  /// side of the level set the point is on
  fn xfem_base(&mut self, v: usize, der: u8, side: i8, out: &mut [f64]) -> Result<()>;
  /// values of im_data v at the point
  fn im_val(&mut self, v: usize, out: &mut [f64]) -> Result<()>;
  /// transformation tr at the point: y the target point (expression), x
  /// the point; strict: a target is required
  fn trans_call(&mut self, tr: usize, y: &[f64], x: &[f64], strict: bool, ders: bool)
  -> Result<()>;
  /// derivative of the target point of tr (contact) times the tests of v
  fn trans_der(&mut self, tr: usize, v: usize, out: &mut [f64]) -> Result<()>;
  fn trans_status(&self, tr: usize) -> usize;
  fn interp_x(&self, tr: usize, out: &mut [f64]) -> Result<()>;
  fn interp_normal(&self, tr: usize, out: &mut [f64]) -> Result<()>;
  fn interp_elt(&self, tr: usize, b: bool, out: &mut [f64]);
  fn interp_field(&mut self, v: usize, der: u8, test: bool, out: &mut [f64]) -> Result<()>;
  /// matrix (column-major, tests of dest x tests of src) of elementary
  /// transformation e on the element
  fn elem_trans(&mut self, e: usize, out: &mut [f64]) -> Result<()>;
  /// the values of an interpolated expression at the point
  fn point_value(&mut self, v: &[f64]) -> Result<()>;
  /// an element vector on the test functions of v
  fn vector(&mut self, v: usize, elem: &[f64]);
  /// an element matrix on the test functions of v1 (rows), v2
  fn matrix(&mut self, v1: usize, v2: usize, elem: &[f64]);
  /// a block elem on the scalar test functions of v1, v2, added on each of
  /// the q components
  fn matrix_opt10(&mut self, v1: usize, v2: usize, q: usize, elem: &[f64]);
  /// condensation: the residual of internal variable v at the point
  fn cached_residual(&mut self, _v: usize, _out: &mut [f64]) -> Result<()> {
    Err(Error::Internal("condensation without a residual".into()))
  }
  /// condensation: K'(q, j) (rows: q at the point, columns: tests of j)
  fn cond_coupling(&mut self, _q: usize, _j: usize, _k: &[f64]) {}
  /// condensation: R'(q) at the point
  fn cond_rq(&mut self, _q: usize, _r: &[f64]) {}
  /// condensation: an element vector of the condensed right-hand side
  fn cond_vector(&mut self, _v: usize, _elem: &[f64]) {}
}

// test only: the compiled instruction sets of this thread, summarized
#[cfg(test)]
thread_local! {
  pub(crate) static COMPILED: std::cell::RefCell<Vec<Summary>> = const { std::cell::RefCell::new(vec![]) };
  pub(crate) static TREES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(vec![]) };
}

/// Test only: instruction counts of an instruction set by kind, the
/// instructions whose result carries the tests of two variables (the
/// size of element matrices) and the fused products by leaf kind
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct Summary {
  pub kinds: std::collections::BTreeMap<String, usize>,
  pub two_tests: usize,
  pub two_tests_kinds: std::collections::BTreeMap<String, usize>,
  pub fused_leaves: std::collections::BTreeMap<String, usize>,
}

impl InstructionSet {
  /// Test only: records the summary of the set in COMPILED
  #[cfg(test)]
  pub(crate) fn record(&self) {
    let name = |d: String| d.split([' ', '{', '(']).next().unwrap().to_string();
    let mut sm = Summary::default();
    for i in self.elt_instructions.iter().chain(&self.instructions) {
      *sm.kinds.entry(name(format!("{i:?}"))).or_default() += 1;
      let two = i.out().is_some_and(|t| {
        matches!(self.shapes[t as usize], Shape::Test { t1, t2, .. } if t1 != NONE && t2 != NONE)
      });
      sm.two_tests += usize::from(two);
      if two {
        *sm
          .two_tests_kinds
          .entry(name(format!("{i:?}")))
          .or_default() += 1;
      }
    }
    for f in &self.fused {
      for leaf in f.leaves() {
        *sm
          .fused_leaves
          .entry(name(format!("{leaf:?}")))
          .or_default() += 1;
      }
    }
    COMPILED.with(|c| c.borrow_mut().push(sm));
  }

  /// GetFEM ga_instruction_condensation_sub at a point: the cluster's
  /// K_QQ inverted (closed forms to order 3), K'_QJ = K_QQ^-1 K_QJ and
  /// R'_Q = K_QQ^-1 R_Q / weight (R_Q is weighted, the blocks are not)
  // out of the instruction loop (kept small for the other instruction sets)
  #[inline(never)]
  fn condense(&self, c: usize, m: &mut ExecState, ext: &mut impl Ext) -> Result<()> {
    let cd = &self.conds[c];
    let nq = cd.q.len();
    let ts: Vec<usize> = cd.rqp.iter().map(|&s| m.range(s).len()).collect();
    let mut off = vec![0; nq + 1];
    for a in 0..nq {
      off[a + 1] = off[a] + ts[a];
    }
    let n = off[nq];
    let mut w = std::mem::take(&mut m.cw);
    w.clear();
    w.resize(2 * n * n + n, 0.0);
    let (inv, rest) = w.split_at_mut(n * n);
    let (r, col) = rest.split_at_mut(n);
    for b in 0..nq {
      for a in 0..nq {
        let s = cd.kqq[a + nq * b];
        if s == NONE {
          continue;
        }
        let blk = &m.buffer[m.range(s)];
        for c2 in 0..ts[b] {
          for c1 in 0..ts[a] {
            inv[off[a] + c1 + n * (off[b] + c2)] = blk[c1 + ts[a] * c2];
          }
        }
      }
    }
    if getfem_geom::geotrans::lu_inverse(inv, n).is_err() {
      let names: Vec<&str> = cd
        .q
        .iter()
        .map(|&q| self.vars[q as usize].as_str())
        .collect();
      m.cw = w;
      return Err(Error::InvalidArgument(format!(
        "condensation: singular block of the internal variables {} at a point",
        names.join(", ")
      )));
    }
    // R'_Q
    for a in 0..nq {
      ext.cached_residual(cd.q[a] as usize, &mut r[off[a]..off[a + 1]])?;
    }
    let wgt = m.coeff;
    for a in 0..nq {
      let o = m.range(cd.rqp[a]).start;
      for c1 in 0..ts[a] {
        let i = off[a] + c1;
        m.buffer[o + c1] = (0..n).map(|k| inv[i + n * k] * r[k]).sum::<f64>() / wgt;
      }
      ext.cond_rq(cd.q[a] as usize, &m.buffer[m.range(cd.rqp[a])]);
    }
    // K'_QJ, column by column of the tests of j
    for (jb, &j) in cd.j.iter().enumerate() {
      let nj = m.range(cd.kqjp[nq * jb]).len() / ts[0].max(1);
      for cj in 0..nj {
        col[..n].fill(0.0);
        for a in 0..nq {
          let s = cd.kqj[a + nq * jb];
          if s != NONE {
            let blk = &m.buffer[m.range(s)];
            col[off[a]..off[a + 1]].copy_from_slice(&blk[ts[a] * cj..ts[a] * (cj + 1)]);
          }
        }
        for a in 0..nq {
          let o = m.range(cd.kqjp[a + nq * jb]).start;
          for c1 in 0..ts[a] {
            let i = off[a] + c1;
            m.buffer[o + c1 + ts[a] * cj] = (0..n).map(|k| inv[i + n * k] * col[k]).sum();
          }
        }
      }
      for a in 0..nq {
        ext.cond_coupling(
          cd.q[a] as usize,
          j as usize,
          &m.buffer[m.range(cd.kqjp[a + nq * jb])],
        );
      }
    }
    m.cw = w;
    Ok(())
  }

  /// t = -sum_a K(i, q_a) K'(q_a, j) (CondenseK) or -sum_a K(i, q_a)
  /// R'(q_a) (CondenseR, jb NONE)
  #[inline(never)]
  fn condense_k(&self, t: u32, k: usize, m: &mut ExecState) {
    let ck = &self.cond_k[k];
    let cd = &self.conds[ck.c as usize];
    let nq = cd.q.len();
    let rt = m.range(t);
    let mut out = std::mem::take(&mut m.pt_elem);
    out.clear();
    out.resize(rt.len(), 0.0);
    for a in 0..nq {
      let s = ck.kiq[a];
      if s == NONE {
        continue;
      }
      let ts = m.range(cd.rqp[a]).len();
      let kiq = &m.buffer[m.range(s)];
      let ni = kiq.len() / ts.max(1);
      let rhs = if ck.jb == NONE {
        cd.rqp[a]
      } else {
        cd.kqjp[a + nq * ck.jb as usize]
      };
      let b = &m.buffer[m.range(rhs)];
      let nj = b.len() / ts.max(1);
      for cj in 0..nj {
        for c in 0..ts {
          let x = b[c + ts * cj];
          if x == 0.0 {
            continue;
          }
          for ii in 0..ni {
            out[ii + ni * cj] -= kiq[ii + ni * c] * x;
          }
        }
      }
    }
    m.buffer[rt].copy_from_slice(&out);
    m.pt_elem = out;
  }
}

impl ExecState {
  /// tensors for the current dims (GetFEM resizes its tensors; here the
  /// buffer is laid out again when the elements change)
  pub fn layout(&mut self, gis: &InstructionSet) {
    self.layouts += 1;
    let mut off = 0u32;
    self.ranges.clear();
    for (k, s) in gis.shapes.iter().enumerate() {
      if let Some(&a) = gis.alias.get(&(k as u32)) {
        let sl = self.ranges[a as usize];
        self.ranges.push(sl);
        continue;
      }
      let len = match *s {
        Shape::Fixed { len, .. } => len as usize,
        Shape::Test { proper, t1, t2 } => {
          let td = |v: u32| {
            if v == NONE {
              1
            } else {
              self.dims[v as usize].tdim()
            }
          };
          proper as usize * td(t1) * td(t2)
        }
        Shape::BaseVal(v) => {
          let d = self.dims[v as usize];
          d.nb * d.td
        }
        Shape::BaseGrad(v) => {
          let d = self.dims[v as usize];
          d.nb * d.td * d.n
        }
        Shape::BaseHess(v) => {
          let d = self.dims[v as usize];
          d.nb * d.td * d.n * d.n
        }
        Shape::Coef(v) => {
          let d = self.dims[v as usize];
          d.nb * (d.qdim / d.td.max(1))
        }
        Shape::ElemMat(a, b) => self.dims[a as usize].tdim() * self.dims[b as usize].tdim(),
      } as u32;
      self.ranges.push(TensorRange { off, len });
      off += len;
    }
    self.buffer.clear();
    self.buffer.resize(off as usize, 0.0);
    for (s, sl) in gis.shapes.iter().zip(&self.ranges) {
      if let Shape::Fixed { len, init } = *s {
        let (o, i, l) = (sl.off as usize, init as usize, len as usize);
        self.buffer[o..o + l].copy_from_slice(&gis.consts[i..i + l]);
      }
    }
  }

  /// the first dimension of a tensor with test functions (GetFEM's
  /// sizes()[0])
  fn first_dim(&self, gis: &InstructionSet, s: u32) -> usize {
    match gis.shapes[s as usize] {
      Shape::Test { t1, t2, .. } => {
        let v = if t1 != NONE { t1 } else { t2 };
        self.dims[v as usize].tdim()
      }
      _ => 1,
    }
  }

  fn range(&self, s: u32) -> std::ops::Range<usize> {
    let s = self.ranges[s as usize];
    s.off as usize..(s.off + s.len) as usize
  }
}

impl InstructionSet {
  fn const_tensor(&mut self, data: &[f64]) -> TensorId {
    let init = self.consts.len() as u32;
    self.consts.extend_from_slice(data);
    self.shapes.push(Shape::Fixed {
      len: data.len() as u32,
      init,
    });
    (self.shapes.len() - 1) as u32
  }

  fn zeros(&mut self, len: usize) -> TensorId {
    self.const_tensor(&vec![0.0; len])
  }

  fn shaped(&mut self, s: Shape) -> TensorId {
    self.shapes.push(s);
    (self.shapes.len() - 1) as u32
  }

  /// a transformation of the instruction set (strict if used outside filters)
  fn tr(&mut self, name: &str) -> Option<u32> {
    let k = self.trans.iter().position(|t| t == name)?;
    if self.in_filter == 0 {
      self.strict[k] = true;
    }
    Some(k as u32)
  }

  fn tr_filter(&mut self, name: &str) -> Option<u32> {
    Some(self.trans.iter().position(|t| t == name)? as u32)
  }

  /// the transformations of the trees, called at the start of the point
  /// instructions (GetFEM ga_instruction_transformation_call)
  fn transformations(&mut self, trees: &[Tree], cx: &CompileCtx) -> Option<()> {
    fn walk(t: &Tree, i: usize, out: &mut Vec<String>) -> Option<()> {
      let n = &t.n[i];
      match n.ty {
        Ty::InterpolateDerivative => {
          out.push(n.interpolate_name_der.clone());
          if !n.interpolate_name.is_empty() {
            out.push(n.interpolate_name.clone());
          }
        }
        Ty::InterpX
        | Ty::InterpNormal
        | Ty::InterpEltK
        | Ty::InterpEltB
        | Ty::InterpolateFilter
        | Ty::Field {
          src: Src::Interp, ..
        } => out.push(n.interpolate_name.clone()),
        _ => {}
      }
      for &c in &n.children {
        walk(t, c, out)?;
      }
      Some(())
    }
    // the transformations read by an expression of transformation first
    fn add(
      gis: &mut InstructionSet,
      name: &str,
      cx: &CompileCtx,
      stack: &mut Vec<String>,
    ) -> Option<()> {
      if gis.trans.iter().any(|t| t == name) {
        return Some(());
      }
      if stack.iter().any(|t| t == name) {
        return None;
      }
      let target = (cx.trans)(name)?;
      if let TransKind::Expr(e) = &target {
        let mut deps = vec![];
        walk(e, e.root?, &mut deps)?;
        stack.push(name.to_string());
        for d in deps {
          add(gis, &d, cx, stack)?;
        }
        stack.pop();
      }
      let y = match &target {
        TransKind::Expr(e) => node(e, e.root?, gis, cx)?,
        TransKind::Neighbor => {
          gis.needs_x = true;
          NONE
        }
        TransKind::Contact => NONE,
      };
      let tr = gis.trans.len() as u32;
      gis.trans.push(name.to_string());
      gis.strict.push(false);
      gis.contact.push(matches!(target, TransKind::Contact));
      gis.trans_ders.push(false);
      gis.instructions.push(Instruction::TransCall { tr, y });
      Some(())
    }
    let mut names = vec![];
    for t in trees {
      walk(t, t.root?, &mut names)?;
    }
    for name in names {
      add(self, &name, cx, &mut vec![])?;
    }
    Some(())
  }

  /// the coefficients of v on the element
  fn coef_of(&mut self, vi: u32) -> TensorId {
    let k = vi as usize;
    if self.coef[k] == NONE {
      let c = self.shaped(Shape::Coef(vi));
      self.coef[k] = c;
      self
        .elt_instructions
        .push(Instruction::SliceDofs { v: vi, c });
    }
    self.coef[k]
  }

  /// the basis of v at the point: values, gradients (Grad, Div), Hessians;
  /// side 1, -1: on that side of the level set the point is on (Xfem_plus,
  /// Xfem_minus)
  fn base_of(&mut self, vi: u32, der: Der, side: i8) -> TensorId {
    let k = vi as usize;
    if side != 0 {
      let (d, shape, code) = match der {
        Der::Val => (Der::Val, Shape::BaseVal(vi), 0),
        Der::Grad | Der::Div => (Der::Grad, Shape::BaseGrad(vi), 1),
        Der::Hess => (Der::Hess, Shape::BaseHess(vi), 2),
      };
      if let Some(&z) = self.xfem_base.get(&(vi, d, side)) {
        return z;
      }
      let z = self.shaped(shape);
      self.xfem_base.insert((vi, d, side), z);
      self.xfem_sides[k] |= if side > 0 { 1 } else { 2 };
      self.needs_hess |= code == 2;
      self.instructions.push(Instruction::XfemBase {
        v: vi,
        z,
        der: code,
        side,
      });
      return z;
    }
    let (cache, shape) = match der {
      Der::Val => (&mut self.base_val, Shape::BaseVal(vi)),
      Der::Grad | Der::Div => (&mut self.base_grad, Shape::BaseGrad(vi)),
      Der::Hess => (&mut self.base_hess, Shape::BaseHess(vi)),
    };
    if cache[k] != NONE {
      return cache[k];
    }
    let z = self.shaped(shape);
    match der {
      Der::Val => {
        self.base_val[k] = z;
        self.instructions.push(Instruction::ValBase { v: vi, z });
      }
      Der::Grad | Der::Div => {
        self.base_grad[k] = z;
        self.instructions.push(Instruction::GradBase { v: vi, z });
      }
      Der::Hess => {
        self.base_hess[k] = z;
        self.needs_hess = true;
        self.instructions.push(Instruction::HessBase { v: vi, z });
      }
    }
    z
  }

  /// the value (der) of v with coefficients c into s
  fn field_val(&mut self, vi: u32, c: u32, der: Der, side: i8, s: u32) {
    let z = self.base_of(vi, der, side);
    self.instructions.push(match der {
      Der::Val => Instruction::Val { t: s, z, c, v: vi },
      Der::Grad => Instruction::Grad { t: s, z, c, v: vi },
      Der::Div => Instruction::Diverg { t: s, z, c, v: vi },
      Der::Hess => Instruction::Hess { t: s, z, c, v: vi },
    });
  }

  /// the test functions (der) of v into s; n3: size of the gradient index
  /// (see fn n3)
  fn copy_test(
    &mut self,
    cx: &CompileCtx,
    vi: u32,
    der: Der,
    side: i8,
    s: u32,
    n3: usize,
  ) -> Option<()> {
    let z = self.base_of(vi, der, side);
    let name = &self.vars[vi as usize];
    let q = cx.defs.var(name)?.qdim() as u8;
    let vectorized = (cx.vectorized)(name);
    match der {
      Der::Val if vectorized => {
        self
          .instructions
          .push(Instruction::CopyVectValBase { t: s, z, v: vi });
        self.set_sp(s, (1, q, 0));
      }
      Der::Val => self
        .instructions
        .push(Instruction::CopyValBase { t: s, z, v: vi }),
      Der::Grad if vectorized => {
        self
          .instructions
          .push(Instruction::CopyVectGradBase { t: s, z, v: vi });
        self.set_sp(s, (2, q, n3 as u8));
      }
      Der::Grad => self
        .instructions
        .push(Instruction::CopyGradBase { t: s, z, v: vi }),
      Der::Div => self
        .instructions
        .push(Instruction::CopyDivergBase { t: s, z, v: vi }),
      Der::Hess => self
        .instructions
        .push(Instruction::CopyHessBase { t: s, z, v: vi }),
    }
    Some(())
  }

  fn var(&mut self, name: &str) -> u32 {
    if let Some(k) = self.vars.iter().position(|v| v == name) {
      return k as u32;
    }
    self.vars.push(name.to_string());
    self.coef.push(NONE);
    self.base_val.push(NONE);
    self.base_grad.push(NONE);
    self.base_hess.push(NONE);
    self.xfem_sides.push(0);
    (self.vars.len() - 1) as u32
  }

  pub fn exec_state(&self) -> ExecState {
    let mut m = ExecState {
      buffer: vec![],
      ranges: vec![],
      dims: vec![VarDims::default(); self.vars.len()],
      coeff: 0.0,
      x: vec![],
      normal: vec![],
      elt_size: 0.0,
      k: vec![],
      b: vec![],
      ipt: 0,
      nbpt: 0,
      acc: 0.0,
      elems: vec![vec![]; self.nb_elems],
      pt_elem: vec![],
      stash: vec![vec![]; self.fused.len()],
      facc: vec![vec![]; self.fused.len()],
      fgemm: vec![vec![]; self.fused.len()],
      flen: vec![0; self.fused.len()],
      fkey: vec![(0, 0); self.fused.len()],
      layouts: 0,
      cw: vec![],
      subs: self.subs.iter().map(GaFunction::exec_state).collect(),
    };
    m.layout(self);
    m
  }

  pub fn run(&self, ins: &[Instruction], m: &mut ExecState, ext: &mut impl Ext) -> Result<()> {
    let mut it = ins.iter();
    while let Some(i) = it.next() {
      match *i {
        // at each point (GetFEM ga_instruction_add_to_coeff)
        Instruction::AddToCoeff { a } => m.acc += m.coeff * m.buffer[m.range(a).start],
        Instruction::X { t } => {
          let r = m.range(t);
          let n = r.len();
          m.buffer[r].copy_from_slice(&m.x[..n]);
        }
        Instruction::XComp { t, k } => {
          let o = m.range(t).start;
          m.buffer[o] = m.x[k as usize];
        }
        Instruction::Add { t, a, b } | Instruction::Sub { t, a, b } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let it = tv.iter_mut().zip(av.iter().zip(bv));
          if let Instruction::Add { .. } = i {
            it.for_each(|(o, (x, y))| *o = x + y);
          } else {
            it.for_each(|(o, (x, y))| *o = x - y);
          }
        }
        Instruction::ScalarMult { t, a, c } | Instruction::ScalarDiv { t, a, c } => {
          let c = m.buffer[m.range(c).start];
          let div = matches!(i, Instruction::ScalarDiv { .. });
          let (rt, ra) = (m.range(t), m.range(a));
          let (tv, av, _) = views(&mut m.buffer, rt, ra.clone(), ra);
          if div {
            for (o, &v) in tv.iter_mut().zip(av) {
              *o = v / c;
            }
          } else {
            for (o, &v) in tv.iter_mut().zip(av) {
              *o = c * v;
            }
          }
        }
        Instruction::SimpleTmult { t, a, b } => {
          // t[i + s1 j] = a[i] b[j] (GetFEM ga_instruction_simple_tmult)
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          for (tj, &y) in tv.chunks_exact_mut(av.len()).zip(bv) {
            for (o, &x) in tj.iter_mut().zip(av) {
              *o = x * y;
            }
          }
        }
        Instruction::Contraction { t, a, b, i } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let i = i as usize;
          let (nn, mm) = (av.len() / i, bv.len() / i);
          // GetFEM: dgemm above 6 x 27 (ga_instruction_contraction)
          if i > 6 && mm * nn * i > 27 {
            gemm(false, true, (mm, nn, i), bv, mm, av, nn, tv, mm);
            continue;
          }
          // a dot product (GetFEM reduc_elem_unrolled for one entry)
          if nn == 1 && mm == 1 {
            let mut s = av[0] * bv[0];
            for (x, y) in av[1..].iter().zip(&bv[1..]) {
              s += x * y;
            }
            tv[0] = s;
            continue;
          }
          // lengths up to 6 unrolled (GetFEM ga_instruction_contraction_unrolled)
          match i {
            1 => contract::<1>(tv, av, bv),
            2 => contract::<2>(tv, av, bv),
            3 => contract::<3>(tv, av, bv),
            4 => contract::<4>(tv, av, bv),
            5 => contract::<5>(tv, av, bv),
            6 => contract::<6>(tv, av, bv),
            _ => {}
          }
          if i <= 6 {
            continue;
          }
          // t[m + M n] = sum_k a[n + N k] b[m + M k], rows of t contiguous
          // (the first term written: no clearing)
          for k in 0..i {
            let bk = &bv[mm * k..mm * (k + 1)];
            for (n, &x) in av[nn * k..nn * (k + 1)].iter().enumerate() {
              let row = tv[mm * n..mm * (n + 1)].iter_mut().zip(bk);
              if k == 0 {
                row.for_each(|(o, y)| *o = x * y);
              } else {
                row.for_each(|(o, y)| *o += x * y);
              }
            }
          }
        }
        Instruction::MatrixMult { t, a, b, j } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let j = j as usize;
          let (mm, kk) = (av.len() / j, bv.len() / j);
          // GetFEM: dgemm above 27 (ga_instruction_matrix_mult)
          if mm * j * kk > 27 {
            gemm(false, false, (mm, kk, j), av, mm, bv, j, tv, mm);
            continue;
          }
          // square 2 x 2 and 3 x 3 unrolled (GetFEM ga_instruction_matrix_mult)
          if mm == j && j == kk && (j == 2 || j == 3) {
            if j == 2 {
              matmul_sq::<2>(tv, av, bv);
            } else {
              matmul_sq::<3>(tv, av, bv);
            }
            continue;
          }
          // columns of t: sum_j b[j + J k] a[:, j] (the first term written)
          for k in 0..kk {
            let tk = &mut tv[mm * k..mm * (k + 1)];
            for (jj, &y) in bv[j * k..j * (k + 1)].iter().enumerate() {
              let col = tk.iter_mut().zip(&av[mm * jj..mm * (jj + 1)]);
              if jj == 0 {
                col.for_each(|(o, x)| *o = x * y);
              } else {
                col.for_each(|(o, x)| *o += x * y);
              }
            }
          }
        }
        Instruction::Func1 { t, a, f } => {
          let f = self.funcs[f as usize];
          let (rt, ra) = (m.range(t), m.range(a));
          for (k, o) in rt.enumerate() {
            m.buffer[o] = f(m.buffer[ra.start + k]);
          }
        }
        Instruction::Copy { t, a } => {
          let ra = m.range(a);
          let o = m.range(t).start;
          m.buffer.copy_within(ra, o);
        }
        Instruction::AddTo { t, a } => {
          let (rt, ra) = (m.range(t), m.range(a));
          let (tv, av, _) = views(&mut m.buffer, rt, ra.clone(), ra);
          tv.iter_mut().zip(av).for_each(|(o, x)| *o += x);
        }
        Instruction::Condense { c } => self.condense(c as usize, m, ext)?,
        Instruction::CondenseK { t, k } | Instruction::CondenseR { t, k } => {
          self.condense_k(t, k as usize, m)
        }
        Instruction::CondVecAssemblyPt { t, v } => {
          let r = m.range(t);
          let c = m.coeff;
          let mut el = std::mem::take(&mut m.pt_elem);
          el.clear();
          el.extend(m.buffer[r].iter().map(|x| x * c));
          ext.cond_vector(v as usize, &el);
          m.pt_elem = el;
        }
        Instruction::Trace { t, a, n } => {
          let (rt, ra) = (m.range(t), m.range(a));
          let n = n as usize;
          let st = rt.len() * (n + 1);
          for k in 0..rt.len() {
            let mut v = 0.0;
            for i in 0..n {
              v += m.buffer[ra.start + k + i * st];
            }
            m.buffer[rt.start + k] = v;
          }
        }
        Instruction::Diverg { t, z, c, v } => {
          let d = m.dims[v as usize];
          let (tt, zz, cc) = (m.range(t).start, m.range(z).start, m.range(c).start);
          let (nb, n, td) = (d.nb, d.n, d.td);
          let mut s = 0.0;
          for k in 0..n {
            for j in 0..nb {
              s += if td == 1 {
                m.buffer[cc + j * n + k] * m.buffer[zz + j + nb * k]
              } else {
                m.buffer[cc + j] * m.buffer[zz + j + nb * (k + td * k)]
              };
            }
          }
          m.buffer[tt] = s;
        }
        Instruction::CopyDivergBase { t, z, v } => {
          let d = m.dims[v as usize];
          let (rt, zz) = (m.range(t), m.range(z).start);
          let (nb, q, td) = (d.nb, d.qdim, d.td);
          m.buffer[rt.clone()].fill(0.0);
          for j in 0..d.n {
            for i in 0..nb {
              if td == 1 {
                m.buffer[rt.start + i * q + j] = m.buffer[zz + i + nb * j];
              } else {
                m.buffer[rt.start + i] += m.buffer[zz + i + nb * (j + td * j)];
              }
            }
          }
        }
        Instruction::Gather { t, a, map } => {
          let (rt, ra) = (m.range(t), m.range(a));
          let map = &self.maps[map as usize];
          let l = rt.len() / map.len();
          for (k, &o) in map.iter().enumerate() {
            let src = ra.start + o as usize * l;
            m.buffer.copy_within(src..src + l, rt.start + k * l);
          }
        }
        Instruction::Func2 { t, a, b, f } => {
          let f = self.funcs2[f as usize];
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (sa, sb) = (ra.len() > 1, rb.len() > 1);
          for (k, o) in rt.enumerate() {
            let x = m.buffer[ra.start + if sa { k } else { 0 }];
            let y = m.buffer[rb.start + if sb { k } else { 0 }];
            m.buffer[o] = f(x, y);
          }
        }
        Instruction::FuncNative { t, a, b, f } => {
          let f = &self.natives[f as usize].0;
          let (rt, ra) = (m.range(t), m.range(a));
          let rb = if b == NONE { None } else { Some(m.range(b)) };
          let sa = ra.len() > 1;
          for (k, o) in rt.enumerate() {
            let x = m.buffer[ra.start + if sa { k } else { 0 }];
            let y = rb.as_ref().map_or(0.0, |rb| {
              m.buffer[rb.start + if rb.len() > 1 { k } else { 0 }]
            });
            m.buffer[o] = f(x, y);
          }
        }
        Instruction::FuncExpr { t, a, b, f } => {
          let sub = &self.subs[f as usize];
          let mut sm = std::mem::take(&mut m.subs[f as usize]);
          let (rt, ra) = (m.range(t), m.range(a));
          let rb = (b != NONE).then(|| m.range(b));
          let ins0 = sm.range(sub.gis.inputs[0]).start;
          let out = sm.range(sub.out).start;
          for k in 0..rt.len() {
            sm.buffer[ins0] = m.buffer[ra.start + if ra.len() > 1 { k } else { 0 }];
            if let Some(rb) = &rb {
              let i1 = sm.range(sub.gis.inputs[1]).start;
              sm.buffer[i1] = m.buffer[rb.start + if rb.len() > 1 { k } else { 0 }];
            }
            sub.gis.run(&sub.gis.instructions, &mut sm, ext)?;
            m.buffer[rt.start + k] = sm.buffer[out];
          }
          m.subs[f as usize] = sm;
        }
        Instruction::CMatrix { t, l } | Instruction::CMatrixTests { t, l } => {
          let comps = &self.maps[l as usize];
          let rt = m.range(t);
          let s = rt.len() / comps.len();
          for (k, &c) in comps.iter().enumerate() {
            let rc = m.range(c);
            for j in 0..s {
              m.buffer[rt.start + k * s + j] =
                m.buffer[rc.start + if rc.len() > 1 { j } else { 0 }];
            }
          }
        }
        Instruction::DotMult { t, a, b } | Instruction::DotDiv { t, a, b } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let s11 = av.len() / bv.len();
          let div = matches!(i, Instruction::DotDiv { .. });
          for (k, &y) in bv.iter().enumerate() {
            for mm in 0..s11 {
              let x = av[mm + s11 * k];
              tv[mm + s11 * k] = if div { x / y } else { x * y };
            }
          }
        }
        Instruction::DotMultSpec { t, a, b } => {
          let s21 = m.first_dim(self, b);
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let s22 = bv.len() / s21;
          let s11 = av.len() / s22;
          let mut it = 0;
          for ii in 0..s22 {
            for n in 0..s21 {
              for mm in 0..s11 {
                tv[it] = av[mm + s11 * ii] * bv[n + s21 * ii];
                it += 1;
              }
            }
          }
        }
        Instruction::Sym { t, a, s: ss } | Instruction::Skew { t, a, s: ss } => {
          let (rt, ra) = (m.range(t), m.range(a));
          let (s1, s2) = ((ss & 0xffff) as usize, (ss >> 16) as usize);
          let s = rt.len() / (s1 * s2);
          let sg = if let Instruction::Sym { .. } = i {
            1.0
          } else {
            -1.0
          };
          for ii in 0..s1 {
            for j in 0..s2 {
              for k in 0..s {
                let x = m.buffer[ra.start + s * (ii + s1 * j) + k];
                let y = m.buffer[ra.start + s * (j + s2 * ii) + k];
                m.buffer[rt.start + s * (ii + s1 * j) + k] = 0.5 * (x + sg * y);
              }
            }
          }
        }
        Instruction::Deviator { t, a, n } => {
          let (rt, ra) = (m.range(t), m.range(a));
          let n = n as usize;
          m.buffer.copy_within(ra.clone(), rt.start);
          let nb = rt.len() / (n * n);
          let s = nb * (n + 1);
          for j in 0..nb {
            let mut tr = 0.0;
            for ii in 0..n {
              tr += m.buffer[ra.start + j + ii * s];
            }
            tr /= n as f64;
            for ii in 0..n {
              m.buffer[rt.start + j + ii * s] -= tr;
            }
          }
        }
        Instruction::SwapInd { t, a, l } => {
          let q = &self.maps[l as usize];
          let (nn1, nn2, ii2, ii3) = (q[0] as usize, q[1] as usize, q[2] as usize, q[3] as usize);
          let (rt, ra) = (m.range(t), m.range(a));
          let ii1 = rt.len() / (nn1 * nn2 * ii2 * ii3);
          let mut it = rt.start;
          for ii in 0..ii3 {
            for j in 0..nn1 {
              for k in 0..ii2 {
                for l in 0..nn2 {
                  let ind =
                    j * ii1 + k * ii1 * nn1 + l * ii1 * nn1 * ii2 + ii * ii1 * nn1 * ii2 * nn2;
                  for mm in 0..ii1 {
                    m.buffer[it] = m.buffer[ra.start + mm + ind];
                    it += 1;
                  }
                }
              }
            }
          }
        }
        Instruction::IndMoveLast { t, a, l } => {
          let q = &self.maps[l as usize];
          let (nn, ii2) = (q[0] as usize, q[1] as usize);
          let (rt, ra) = (m.range(t), m.range(a));
          let ii1 = rt.len() / (nn * ii2);
          let mut it = rt.start;
          for ii in 0..nn {
            for j in 0..ii2 {
              let ind = ii * ii1 + j * ii1 * nn;
              for k in 0..ii1 {
                m.buffer[it] = m.buffer[ra.start + k + ind];
                it += 1;
              }
            }
          }
        }
        Instruction::Contract11 { t, a, l } => {
          let q = &self.maps[l as usize];
          let (nn, ii2, ii3) = (q[0] as usize, q[1] as usize, q[2] as usize);
          let (rt, ra) = (m.range(t), m.range(a));
          let ii1 = ra.len() / (nn * nn * ii2 * ii3);
          let mut it = rt.start;
          for ii in 0..ii3 {
            for j in 0..ii2 {
              for k in 0..ii1 {
                let pre = k + j * ii1 * nn + ii * ii1 * nn * ii2 * nn;
                let mut v = 0.0;
                for n in 0..nn {
                  v += m.buffer[ra.start + pre + n * ii1 + n * ii1 * nn * ii2];
                }
                m.buffer[it] = v;
                it += 1;
              }
            }
          }
        }
        Instruction::Contract21 { t, a, b, l } => {
          let q = &self.maps[l as usize];
          let (nn, ii1, ii2, ii3, ii4) = (
            q[0] as usize,
            q[1] as usize,
            q[2] as usize,
            q[3] as usize,
            q[4] as usize,
          );
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let ift1 = av.len() / (nn * ii1 * ii2);
          let ift2 = bv.len() / (nn * ii3 * ii4);
          let mut it = 0;
          for ii in 0..ii4 {
            for j in 0..ii3 {
              for k in 0..ii2 {
                for l in 0..ii1 {
                  for pp in 0..ift2 {
                    for qq in 0..ift1 {
                      let ind1 = qq + l * ift1 + k * ift1 * ii1 * nn;
                      let ind2 = pp + j * ift2 + ii * ift2 * ii3 * nn;
                      let mut v = 0.0;
                      for n in 0..nn {
                        v += av[ind1 + n * ift1 * ii1] * bv[ind2 + n * ift2 * ii3];
                      }
                      tv[it] = v;
                      it += 1;
                    }
                  }
                }
              }
            }
          }
        }
        Instruction::Cross { t, a, b } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, x, y) = views(&mut m.buffer, rt, ra, rb);
          tv[0] = x[1] * y[2] - x[2] * y[1];
          tv[1] = x[2] * y[0] - x[0] * y[2];
          tv[2] = x[0] * y[1] - x[1] * y[0];
        }
        Instruction::CrossTf { t, a, b, inv } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, x, y) = views(&mut m.buffer, rt, ra, rb);
          let (n1, n2) = (x.len() / 3, y.len() / 3);
          let nn = n1 * n2;
          let s = if inv != 0 { -1.0 } else { 1.0 };
          let mut it = 0;
          for ii in 0..n2 {
            for j in 0..n1 {
              tv[it] = s * (x[j + n1] * y[ii + 2 * n2] - x[j + 2 * n1] * y[ii + n2]);
              tv[it + nn] = s * (x[j + 2 * n1] * y[ii] - x[j] * y[ii + 2 * n2]);
              tv[it + 2 * nn] = s * (x[j] * y[ii + n2] - x[j + n1] * y[ii]);
              it += 1;
            }
          }
        }
        Instruction::HessBase { v, z } => {
          let r = m.range(z);
          ext.hess_base(v as usize, &mut m.buffer[r])?;
        }
        Instruction::ImVal { t, v } => {
          let r = m.range(t);
          ext.im_val(v as usize, &mut m.buffer[r])?;
        }
        Instruction::TransCall { tr, y } => {
          let yv: &[f64] = if y == NONE {
            &[]
          } else {
            &m.buffer[m.range(y)]
          };
          let k = tr as usize;
          ext.trans_call(k, yv, &m.x, self.strict[k], self.trans_ders[k])?;
        }
        Instruction::TransDer { t, tr, v } => {
          let r = m.range(t);
          ext.trans_der(tr as usize, v as usize, &mut m.buffer[r])?;
        }
        Instruction::InterpX { t, tr } => {
          let r = m.range(t);
          ext.interp_x(tr as usize, &mut m.buffer[r])?;
        }
        Instruction::InterpElt { t, tr, b } => {
          let r = m.range(t);
          ext.interp_elt(tr as usize, b != 0, &mut m.buffer[r]);
        }
        Instruction::InterpNormal { t, tr } => {
          let r = m.range(t);
          ext.interp_normal(tr as usize, &mut m.buffer[r])?;
        }
        Instruction::InterpField { t, v, der, test } => {
          let r = m.range(t);
          ext.interp_field(v as usize, der, test != 0, &mut m.buffer[r])?;
        }
        Instruction::Filter { t, tr, st, len } => {
          // st NONE: every point with a target (GetFEM pt_type -1)
          let status = ext.trans_status(tr as usize);
          let pass = if st == NONE {
            status != 0
          } else {
            status == st as usize
          };
          if !pass {
            let r = m.range(t);
            m.buffer[r].fill(0.0);
            if len > 0 {
              it.nth(len as usize - 1);
            }
          }
        }
        Instruction::EltTrans { m: mm, e } => {
          let r = m.range(mm);
          ext.elem_trans(e as usize, &mut m.buffer[r])?;
        }
        Instruction::ElemCoef { t, m: mm, c } => {
          let (rt, rm, rc) = (m.range(t), m.range(mm), m.range(c));
          let nr = rt.len();
          for j in 0..nr {
            m.buffer[rt.start + j] = 0.0;
          }
          for (k, ck) in rc.clone().enumerate() {
            let x = m.buffer[ck];
            for j in 0..nr {
              m.buffer[rt.start + j] += m.buffer[rm.start + j + nr * k] * x;
            }
          }
        }
        Instruction::ElemTest { t, a, m: mm, d } => {
          let (rt, ra, rm) = (m.range(t), m.range(a), m.range(mm));
          let (nd, ns) = (ra.len(), rt.len());
          let nr = m.dims[d as usize].tdim();
          let np = nd / nr;
          let nc = ns / np;
          for q in 0..np {
            for k in 0..nc {
              let mut s = 0.0;
              for j in 0..nr {
                s += m.buffer[rm.start + j + nr * k] * m.buffer[ra.start + j + nr * q];
              }
              m.buffer[rt.start + k + nc * q] = s;
            }
          }
        }
        Instruction::PointValue { a } => {
          let r = m.range(a);
          ext.point_value(&m.buffer[r])?;
        }
        Instruction::VecAssemblyPt { t, v } => {
          let r = m.range(t);
          let c = m.coeff;
          let mut el = std::mem::take(&mut m.pt_elem);
          el.clear();
          el.extend(m.buffer[r].iter().map(|x| x * c));
          ext.vector(v as usize, &el);
          m.pt_elem = el;
        }
        Instruction::MatAssemblyPt { t, v1, v2 } => {
          let r = m.range(t);
          let c = m.coeff;
          let mut el = std::mem::take(&mut m.pt_elem);
          el.clear();
          el.extend(m.buffer[r].iter().map(|x| x * c));
          ext.matrix(v1 as usize, v2 as usize, &el);
          m.pt_elem = el;
        }
        Instruction::EltSize { t } => {
          let o = m.range(t).start;
          m.buffer[o] = m.elt_size;
        }
        Instruction::EltK { t } | Instruction::EltB { t } => {
          let r = m.range(t);
          let n = r.len();
          let src = if let Instruction::EltK { .. } = i {
            &m.k
          } else {
            &m.b
          };
          m.buffer[r].copy_from_slice(&src[..n]);
        }
        Instruction::Nonlin { t, a, b, op } => {
          // arguments and result in the buffer, no allocation (GetFEM:
          // ga_instruction_eval_OP)
          let nl = &self.nonlins[op as usize];
          let ra = m.range(a);
          let rb = if b == NONE { ra.clone() } else { m.range(b) };
          let rt = m.range(t);
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let na = 1 + (b != NONE) as usize;
          let args: [&[f64]; 2] = [av, bv];
          let sh: [&[usize]; 2] = [&nl.shapes[0], nl.shapes.get(1).map_or(&[][..], |s| s)];
          let (args, sh) = (&args[..na], &sh[..na]);
          match nl.der {
            (0, _) => nl.op.value(args, sh, tv),
            (d, 0) => nl.op.derivative(args, sh, d - 1, tv),
            (d, e) => nl.op.second_derivative(args, sh, d - 1, e - 1, tv),
          }
          .map_err(Error::InvalidArgument)?;
        }
        Instruction::SliceDofs { v, c } => {
          let r = m.range(c);
          ext.slice_dofs(v as usize, &mut m.buffer[r]);
        }
        Instruction::ValBase { v, z } => {
          let r = m.range(z);
          ext.val_base(v as usize, &mut m.buffer[r]);
        }
        Instruction::GradBase { v, z } => {
          let r = m.range(z);
          ext.grad_base(v as usize, &mut m.buffer[r]);
        }
        Instruction::XfemBase { v, z, der, side } => {
          let r = m.range(z);
          ext.xfem_base(v as usize, der, side, &mut m.buffer[r])?;
        }
        Instruction::Val { t, z, c, v }
        | Instruction::Grad { t, z, c, v }
        | Instruction::Hess { t, z, c, v } => {
          let d = m.dims[v as usize];
          let (tt, zz, cc) = (m.range(t).start, m.range(z).start, m.range(c).start);
          let (nb, q, td) = (d.nb, d.qdim, d.td);
          let qm = q / td;
          // GetFEM's ga_instruction_val/grad/hess: t[r + q' td + k qdim] =
          // sum_j coeff[j Qmult + q'] Z[j + nb (r + td k)]
          let nn = match i {
            Instruction::Val { .. } => 1,
            Instruction::Grad { .. } => d.n,
            _ => d.n * d.n,
          };
          if td == 1 {
            let (rt, rc, rz) = (m.range(t), m.range(c), m.range(z));
            let (tv, cv, zv) = views(&mut m.buffer, rt, rc, rz);
            let zv = &zv[..nb * nn];
            match q {
              1 => val_scalar::<1>(tv, cv, zv, nb),
              2 => val_scalar::<2>(tv, cv, zv, nb),
              3 => val_scalar::<3>(tv, cv, zv, nb),
              _ => val_scalar_q(tv, cv, zv, nb, q),
            }
            continue;
          }
          for k in 0..nn {
            for qq in 0..qm {
              for rr in 0..td {
                let mut s = 0.0;
                for j in 0..nb {
                  s += m.buffer[cc + j * qm + qq] * m.buffer[zz + j + nb * (rr + td * k)];
                }
                m.buffer[tt + rr + qq * td + k * q] = s;
              }
            }
          }
        }
        Instruction::CopyValBase { t, z, v }
        | Instruction::CopyGradBase { t, z, v }
        | Instruction::CopyHessBase { t, z, v } => {
          let d = m.dims[v as usize];
          let (rt, rz) = (m.range(t), m.range(z));
          let (nb, td) = (d.nb, d.td);
          let qm = d.qdim / td;
          if qm == 1 {
            let zz = rz.start;
            m.buffer.copy_within(zz..zz + rz.len(), rt.start);
          } else {
            // t(i Q + j, j td + k, l) = Z(i, k, l): components as the
            // values (GetFEM's copy_val_base: k Q + j, inconsistent)
            let nl = match i {
              Instruction::CopyValBase { .. } => 1,
              Instruction::CopyGradBase { .. } => d.n,
              _ => d.n * d.n,
            };
            let s = nb * qm;
            m.buffer[rt.clone()].fill(0.0);
            for l in 0..nl {
              for k in 0..td {
                for ii in 0..nb {
                  let zv = m.buffer[rz.start + ii + nb * (k + td * l)];
                  for j in 0..qm {
                    m.buffer[rt.start + ii * qm + j + s * (j * td + k + qm * td * l)] = zv;
                  }
                }
              }
            }
          }
        }
        Instruction::Normal { t } => {
          let r = m.range(t);
          let n = r.len();
          m.buffer[r].copy_from_slice(&m.normal[..n]);
        }
        Instruction::Transpose { t, a, jk, i } => {
          let (rt, ra) = (m.range(t), m.range(a));
          let (j, k, ii) = ((jk & 0xffff) as usize, (jk >> 16) as usize, i as usize);
          let l = ra.len() / (j * k * ii);
          let mut it = rt.start;
          for i in 0..ii {
            for jj in 0..j {
              for kk in 0..k {
                let s3 = ra.start + i * j * k * l + jj * l + kk * j * l;
                m.buffer.copy_within(s3..s3 + l, it);
                it += l;
              }
            }
          }
        }
        Instruction::SpecContraction { t, a, b, nn } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let nn = nn as usize;
          let s1 = m.first_dim(self, a);
          let s11 = ra.len() / (s1 * nn);
          let s111 = s1 * s11;
          let s2 = m.first_dim(self, b);
          let mut it = rt.start;
          for i in 0..s11 {
            for n in 0..s2 {
              for mm in 0..s1 {
                let mut v = 0.0;
                for j in 0..nn {
                  v +=
                    m.buffer[ra.start + mm + i * s1 + j * s111] * m.buffer[rb.start + n + j * s2];
                }
                m.buffer[it] = v;
                it += 1;
              }
            }
          }
        }
        Instruction::Spec2Contraction { t, a, b, nn } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let nn = nn as usize;
          let s1 = m.first_dim(self, a);
          let s11 = ra.len() / (s1 * nn);
          let s111 = s1 * s11;
          let s2 = m.first_dim(self, b);
          let s22 = rb.len() / (s2 * nn);
          let s222 = s2 * s22;
          let mut it = rt.start;
          for j in 0..s22 {
            for i in 0..s11 {
              for mm in 0..s1 {
                for n in 0..s2 {
                  let mut v = 0.0;
                  for k in 0..nn {
                    v += m.buffer[ra.start + mm + i * s1 + k * s111]
                      * m.buffer[rb.start + n + j * s2 + k * s222];
                  }
                  m.buffer[it] = v;
                  it += 1;
                }
              }
            }
          }
        }
        // GetFEM with BLAS: dgemm per (i, k) block, per k block
        // (ga_instruction_matrix_mult_spec, _spec2)
        Instruction::MatrixMultSpec { t, a, b, jik }
        | Instruction::MatrixMultSpec2 { t, a, b, jik } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let (j, ii, k) = (
            (jik & 0x3ff) as usize,
            ((jik >> 10) & 0x3ff) as usize,
            (jik >> 20) as usize,
          );
          let mi_ = av.len() / j;
          let nj = bv.len() / k;
          let n_ = nj / j;
          if let Instruction::MatrixMultSpec { .. } = i {
            let m_ = mi_ / ii;
            let mn = m_ * n_;
            for kk in 0..k {
              for i2 in 0..ii {
                let o = mn * (i2 + ii * kk);
                gemm(
                  false,
                  true,
                  (m_, n_, j),
                  &av[m_ * i2..],
                  mi_,
                  &bv[nj * kk..],
                  n_,
                  &mut tv[o..o + mn],
                  m_,
                );
              }
            }
          } else {
            let nmi = n_ * mi_;
            for kk in 0..k {
              let o = nmi * kk;
              gemm(
                false,
                true,
                (n_, mi_, j),
                &bv[nj * kk..],
                n_,
                av,
                mi_,
                &mut tv[o..o + nmi],
                n_,
              );
            }
          }
        }
        Instruction::SpecTmult { t, a, b, ij } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let (ii, j) = ((ij & 0xffff) as usize, ((ij >> 16) & 0x7fff) as usize);
          let swap = ij >> 31 != 0;
          let (m_, n_) = (av.len() / ii, bv.len() / j);
          for jj in 0..j {
            let bj = &bv[n_ * jj..n_ * (jj + 1)];
            for i2 in 0..ii {
              let ai = &av[m_ * i2..m_ * (i2 + 1)];
              let blk = &mut tv[m_ * n_ * (i2 + ii * jj)..m_ * n_ * (i2 + ii * jj + 1)];
              if swap {
                // test1 first when child0 carries test2: t[n + N m]
                for (mm, &x) in ai.iter().enumerate() {
                  for (o, &c) in blk[n_ * mm..n_ * (mm + 1)].iter_mut().zip(bj) {
                    *o = x * c;
                  }
                }
              } else {
                for (n, &c) in bj.iter().enumerate() {
                  for (o, &x) in blk[m_ * n..m_ * (n + 1)].iter_mut().zip(ai) {
                    *o = x * c;
                  }
                }
              }
            }
          }
        }
        Instruction::Spec2Tmult { t, a, b } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let ii = ra.len();
          let m_ = m.first_dim(self, b);
          let j = rb.len() / m_;
          let mut it = rt.start;
          for jj in 0..j {
            for i2 in 0..ii {
              for mm in 0..m_ {
                m.buffer[it] = m.buffer[ra.start + i2] * m.buffer[rb.start + mm + m_ * jj];
                it += 1;
              }
            }
          }
        }
        Instruction::CopyVectValBase { t, z, v } | Instruction::CopyVectGradBase { t, z, v } => {
          let d = m.dims[v as usize];
          let (rt, rz) = (m.range(t), m.range(z));
          let (nb, q) = (d.nb, d.qdim);
          let nl = if let Instruction::CopyVectGradBase { .. } = i {
            d.n
          } else {
            1
          };
          // t(i q + j, j + q l) = Z(i, l), the rest stays zero
          // column j + q l of t (s rows): Z(i, l) at rows i q + j
          let (tv, zv, _) = views(&mut m.buffer, rt, rz.clone(), rz);
          match q {
            2 => copy_vect::<2>(tv, zv, nb, nl),
            3 => copy_vect::<3>(tv, zv, nb, nl),
            _ => {
              let s = nb * q;
              for (c, col) in tv.chunks_exact_mut(s).take(q * nl).enumerate() {
                let (j, l) = (c % q, c / q);
                for (y, &x) in col.chunks_exact_mut(q).zip(&zv[nb * l..nb * (l + 1)]) {
                  y[j] = x;
                }
              }
            }
          }
        }
        Instruction::ContractionOpt11 { t, a, b, nn } => {
          // the nonzeros only (the rest stays zero, GetFEM opt1_1)
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let nn = nn as usize;
          // column ii nn + k of t (s2 rows): a[ii nn] b[j nn] at rows j nn + k
          match nn {
            1 => opt11::<1>(tv, av, bv),
            2 => opt11::<2>(tv, av, bv),
            3 => opt11::<3>(tv, av, bv),
            _ => {
              let s2 = bv.len() / nn;
              for (c, col) in tv.chunks_exact_mut(s2).enumerate() {
                let (ii, k) = (c / nn, c % nn);
                let x = av[ii * nn];
                for (y, b) in col.chunks_exact_mut(nn).zip(bv.chunks_exact(nn)) {
                  y[k] = x * b[0];
                }
              }
            }
          }
        }
        Instruction::ContractionOpt01 { t, a, b, nn } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let nn = nn as usize;
          if nn == 2 || nn == 3 {
            if nn == 2 {
              opt01::<2>(tv, av, bv);
            } else {
              opt01::<3>(tv, av, bv);
            }
            continue;
          }
          let s1 = av.len() / nn;
          let s2n = bv.len() / nn / nn;
          let mut it = tv.chunks_exact_mut(nn);
          for ii in 0..s1 {
            for b in bv.chunks_exact(nn).take(s2n) {
              let o = it.next().unwrap();
              for (k, y) in o.iter_mut().enumerate() {
                *y = av[ii + k * s1] * b[0];
              }
            }
          }
        }
        Instruction::ContractionOpt02 { t, a, b, nq } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let (n, q) = ((nq & 0xffff) as usize, (nq >> 16) as usize);
          let nn = n * q;
          let (s1, s2) = (av.len() / nn, bv.len() / nn);
          let (s2q, s1qq, s2qq) = (s2 / q, s1 * q, s2 * q);
          // t[l + q j + s2 i] = sum_m a[i + s1 l + s1 q m] b[j q + s2 q m]
          match q {
            2 => match n {
              2 => opt02n::<2, 2>(tv, av, bv),
              3 => opt02n::<3, 2>(tv, av, bv),
              _ => opt02::<2>(tv, av, bv, n),
            },
            3 => match n {
              2 => opt02n::<2, 3>(tv, av, bv),
              3 => opt02n::<3, 3>(tv, av, bv),
              _ => opt02::<3>(tv, av, bv, n),
            },
            _ => {
              tv.fill(0.0);
              for ii in 0..s1 {
                let row = &mut tv[s2 * ii..s2 * (ii + 1)];
                for mm in 0..n {
                  let al = &av[ii + s1qq * mm..];
                  for j in 0..s2q {
                    let y = bv[j * q + s2qq * mm];
                    for (l, o) in row[j * q..(j + 1) * q].iter_mut().enumerate() {
                      *o += al[s1 * l] * y;
                    }
                  }
                }
              }
            }
          }
        }
        Instruction::ContractionOpt20 { t, a, b, nq } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let (n, q) = ((nq & 0xffff) as usize, (nq >> 16) as usize);
          let nn = n * q;
          let (s1, s2) = (av.len() / nn, bv.len() / nn);
          let (s1q, s1qq, s2qq) = (s1 / q, s1 * q, s2 * q);
          // t[j + s2 (l + q i)] = sum_m a[i q + s1 q m] b[l s2 + j + s2 q m]
          // (the first term written)
          if n == 2 || n == 3 {
            if n == 2 {
              opt20::<2>(tv, av, bv, q);
            } else {
              opt20::<3>(tv, av, bv, q);
            }
            continue;
          }
          for ii in 0..s1q {
            for l in 0..q {
              let row = &mut tv[s2 * (l + q * ii)..s2 * (l + q * ii + 1)];
              for mm in 0..n {
                let x = av[ii * q + s1qq * mm];
                let br = &bv[l * s2 + s2qq * mm..l * s2 + s2qq * mm + s2];
                let it = row.iter_mut().zip(br);
                if mm == 0 {
                  it.for_each(|(o, y)| *o = x * y);
                } else {
                  it.for_each(|(o, y)| *o += x * y);
                }
              }
            }
          }
        }
        Instruction::ContractionOpt22 { t, a, b, nq } => {
          let (rt, ra, rb) = (m.range(t), m.range(a), m.range(b));
          let (tv, av, bv) = views(&mut m.buffer, rt, ra, rb);
          let (n, q) = ((nq & 0xffff) as usize, (nq >> 16) as usize);
          let (s1, s2) = (av.len() / (n * q), bv.len() / (n * q));
          // the nonzeros only: t[j q + k + s2 (i q + k)] = sum_m a[i q + s1 q
          // m] b[j q + s2 q m] for each component k
          for i in 0..s1 / q {
            for j in 0..s2 / q {
              let x: f64 = (0..n)
                .map(|mm| av[i * q + s1 * q * mm] * bv[j * q + s2 * q * mm])
                .sum();
              for k in 0..q {
                tv[j * q + k + s2 * (i * q + k)] = x;
              }
            }
          }
        }
        Instruction::MatAssemblyOpt10 { t, v1, v2, e } => {
          let r = m.range(t);
          let (d1, d2) = (m.dims[v1 as usize], m.dims[v2 as usize]);
          let q = d1.qdim;
          let (ss1, ss2) = (d1.nb, d2.nb);
          let s1q = q * d1.tdim();
          let (c, e) = (m.coeff, e as usize);
          let mut el = std::mem::take(&mut m.elems[e]);
          el.resize(ss1 * ss2, 0.0);
          let tv = &m.buffer[r];
          if m.ipt == 0 {
            for (j, ej) in el.chunks_exact_mut(ss1).enumerate() {
              for (o, x) in ej
                .iter_mut()
                .zip(tv[j * s1q..j * s1q + ss1 * q].chunks_exact(q))
              {
                *o = x[0] * c;
              }
            }
          } else {
            for (j, ej) in el.chunks_exact_mut(ss1).enumerate() {
              for (o, x) in ej
                .iter_mut()
                .zip(tv[j * s1q..j * s1q + ss1 * q].chunks_exact(q))
              {
                *o += x[0] * c;
              }
            }
          }
          if m.ipt + 1 == m.nbpt {
            ext.matrix_opt10(v1 as usize, v2 as usize, q, &el);
          }
          m.elems[e] = el;
        }
        Instruction::FusedAssembly { f } => self.fused_assembly(f as usize, m, ext),
        Instruction::MatAssembly { t, v1, v2, e } => {
          let r = m.range(t);
          let (c, e) = (m.coeff, e as usize);
          let mut el = std::mem::take(&mut m.elems[e]);
          el.resize(r.len(), 0.0);
          if m.ipt == 0 {
            for (a, x) in el.iter_mut().zip(&m.buffer[r]) {
              *a = x * c;
            }
          } else {
            for (a, x) in el.iter_mut().zip(&m.buffer[r]) {
              *a += x * c;
            }
          }
          if m.ipt + 1 == m.nbpt {
            ext.matrix(v1 as usize, v2 as usize, &el);
          }
          m.elems[e] = el;
        }
        Instruction::VecAssembly { t, v, e } => {
          let r = m.range(t);
          let (c, e) = (m.coeff, e as usize);
          let mut el = std::mem::take(&mut m.elems[e]);
          el.resize(r.len(), 0.0);
          if m.ipt == 0 {
            for (a, x) in el.iter_mut().zip(&m.buffer[r]) {
              *a = x * c;
            }
          } else {
            for (a, x) in el.iter_mut().zip(&m.buffer[r]) {
              *a += x * c;
            }
          }
          if m.ipt + 1 == m.nbpt {
            ext.vector(v as usize, &el);
          }
          m.elems[e] = el;
        }
      }
    }
    Ok(())
  }
}

/// t = a b, N x N (the terms in the order of the general loop)
fn matmul_sq<const N: usize>(tv: &mut [f64], av: &[f64], bv: &[f64]) {
  for k in 0..N {
    for i in 0..N {
      let mut s = av[i] * bv[N * k];
      for j in 1..N {
        s += av[i + N * j] * bv[j + N * k];
      }
      tv[i + N * k] = s;
    }
  }
}

/// opt0_1 for N (GetFEM ga_instruction_contraction_opt0_1_unrolled):
/// t[k + N (j + s2n i)] = a[i + s1 k] b[j N]
fn opt01<const N: usize>(tv: &mut [f64], av: &[f64], bv: &[f64]) {
  let s1 = av.len() / N;
  let s2n = bv.len() / N / N;
  let (ta, _) = tv.as_chunks_mut::<N>();
  let (ba, _) = bv.as_chunks::<N>();
  for (ii, row) in ta.chunks_exact_mut(s2n).enumerate().take(s1) {
    let ai: [f64; N] = std::array::from_fn(|k| av[ii + s1 * k]);
    for (o, b) in row.iter_mut().zip(ba) {
      let y = b[0];
      for k in 0..N {
        o[k] = ai[k] * y;
      }
    }
  }
}

/// values (gradients, Hessians) of a field of Q copies of a scalar element:
/// t[q + Q k] = sum_j c[j Q + q] Z[j + nb k] (GetFEM ga_instruction_val)
fn val_scalar<const Q: usize>(tv: &mut [f64], cv: &[f64], zv: &[f64], nb: usize) {
  let (ca, _) = cv.as_chunks::<Q>();
  let (ta, _) = tv.as_chunks_mut::<Q>();
  for (o, zk) in ta.iter_mut().zip(zv.chunks_exact(nb)) {
    let mut s = [0.0; Q];
    for (c, &z) in ca.iter().zip(zk) {
      for q in 0..Q {
        s[q] += c[q] * z;
      }
    }
    *o = s;
  }
}

/// val_scalar for any q: the sums along j by blocks of 4 components
/// (fixed width: vectorized), then the rest
fn val_scalar_q(tv: &mut [f64], cv: &[f64], zv: &[f64], nb: usize, q: usize) {
  for (o, zk) in tv.chunks_exact_mut(q).zip(zv.chunks_exact(nb)) {
    let (o4, rest) = o.as_chunks_mut::<4>();
    for (b, ob) in o4.iter_mut().enumerate() {
      let mut s = [0.0; 4];
      for (j, &z) in zk.iter().enumerate() {
        let c = &cv[j * q + 4 * b..][..4];
        for i in 0..4 {
          s[i] += c[i] * z;
        }
      }
      *ob = s;
    }
    let i0 = q - rest.len();
    for (i, s) in rest.iter_mut().enumerate() {
      *s = zk
        .iter()
        .enumerate()
        .fold(0.0, |s, (j, &z)| s + cv[j * q + i0 + i] * z);
    }
  }
}

/// t[m + M n] = sum_k a[n + N k] b[m + M k] for the length I: per row n,
/// the I factors of a, then the sums along m (the same order in k)
fn contract<const I: usize>(tv: &mut [f64], av: &[f64], bv: &[f64]) {
  let (nn, mm) = (av.len() / I, bv.len() / I);
  if mm == 0 {
    return;
  }
  let bk: [&[f64]; I] = std::array::from_fn(|k| &bv[mm * k..mm * (k + 1)]);
  for (n, row) in tv.chunks_exact_mut(mm).enumerate().take(nn) {
    let ak: [f64; I] = std::array::from_fn(|k| av[n + nn * k]);
    match I {
      // one pass, no index checks
      2 => {
        for (o, (&y0, &y1)) in row.iter_mut().zip(bk[0].iter().zip(bk[1])) {
          *o = ak[0] * y0 + ak[1] * y1;
        }
      }
      3 => {
        let b12 = bk[1].iter().zip(bk[2]);
        for (o, (&y0, (&y1, &y2))) in row.iter_mut().zip(bk[0].iter().zip(b12)) {
          *o = ak[0] * y0 + ak[1] * y1 + ak[2] * y2;
        }
      }
      _ => {
        for (m, o) in row.iter_mut().enumerate() {
          let mut s = ak[0] * bk[0][m];
          for k in 1..I {
            s += ak[k] * bk[k][m];
          }
          *o = s;
        }
      }
    }
  }
}

/// opt2_0 for n = N (GetFEM opt2_0_dunrolled): each row of t in one pass,
/// sum_m x_m b_m in the order of m
fn opt20<const N: usize>(tv: &mut [f64], av: &[f64], bv: &[f64], q: usize) {
  let nn = N * q;
  let (s1, s2) = (av.len() / nn, bv.len() / nn);
  let (s1q, s1qq, s2qq) = (s1 / q, s1 * q, s2 * q);
  for ii in 0..s1q {
    for l in 0..q {
      let row = &mut tv[s2 * (l + q * ii)..s2 * (l + q * ii + 1)];
      let x: [f64; N] = std::array::from_fn(|m| av[ii * q + s1qq * m]);
      let br: [&[f64]; N] = std::array::from_fn(|m| &bv[l * s2 + s2qq * m..l * s2 + s2qq * m + s2]);
      for (j, o) in row.iter_mut().enumerate() {
        let mut s = x[0] * br[0][j];
        for m in 1..N {
          s += x[m] * br[m][j];
        }
        *o = s;
      }
    }
  }
}

/// opt0_2 for q = Q: row i of t as arrays of Q, array j = sum_m
/// (a[i + s1 l + s1 Q m])_l b[j Q + s2 Q m] (the first term written)
fn opt02<const Q: usize>(tv: &mut [f64], av: &[f64], bv: &[f64], n: usize) {
  let nn = n * Q;
  let (s1, s2) = (av.len() / nn, bv.len() / nn);
  let (s1qq, s2qq) = (s1 * Q, s2 * Q);
  for (ii, row) in tv.chunks_exact_mut(s2).enumerate() {
    let (ra, _) = row.as_chunks_mut::<Q>();
    for mm in 0..n {
      let al: [f64; Q] = std::array::from_fn(|l| av[ii + s1 * l + s1qq * mm]);
      let (bm, _) = bv[s2qq * mm..s2qq * mm + s2].as_chunks::<Q>();
      for (o, b) in ra.iter_mut().zip(bm) {
        let y = b[0];
        if mm == 0 {
          for l in 0..Q {
            o[l] = al[l] * y;
          }
        } else {
          for l in 0..Q {
            o[l] += al[l] * y;
          }
        }
      }
    }
  }
}

/// opt02 for n = N too (GetFEM's contraction_opt0_2_dunrolled): each entry
/// summed over m in a register, the same order of operations
fn opt02n<const N: usize, const Q: usize>(tv: &mut [f64], av: &[f64], bv: &[f64]) {
  let nn = N * Q;
  let (s1, s2) = (av.len() / nn, bv.len() / nn);
  let (s1qq, s2qq) = (s1 * Q, s2 * Q);
  for (ii, row) in tv.chunks_exact_mut(s2).enumerate() {
    let al: [[f64; Q]; N] =
      std::array::from_fn(|m| std::array::from_fn(|l| av[ii + s1 * l + s1qq * m]));
    let (ra, _) = row.as_chunks_mut::<Q>();
    for (j, o) in ra.iter_mut().enumerate() {
      let bm: [f64; N] = std::array::from_fn(|m| bv[j * Q + s2qq * m]);
      for l in 0..Q {
        let mut s = al[0][l] * bm[0];
        for m in 1..N {
          s += al[m][l] * bm[m];
        }
        o[l] = s;
      }
    }
  }
}

/// opt1_1 for N (1 to 3; GetFEM unrolls too): t viewed as arrays of N,
/// column c = ii N + k: a[ii N] b[j N] at entry k of array j of the column
fn opt11<const N: usize>(tv: &mut [f64], av: &[f64], bv: &[f64]) {
  let (ta, _) = tv.as_chunks_mut::<N>();
  let (ba, _) = bv.as_chunks::<N>();
  let ss2 = ba.len() / N;
  for (c, col) in ta.chunks_exact_mut(ss2).enumerate() {
    let (ii, k) = (c / N, c % N);
    let x = av[ii * N];
    for (y, b) in col.iter_mut().zip(ba) {
      y[k] = x * b[0];
    }
  }
}

/// the vectorized copy for q = N: t(i N + j, j + N l) = Z(i, l)
fn copy_vect<const N: usize>(tv: &mut [f64], zv: &[f64], nb: usize, nl: usize) {
  // column j + N l (N nb rows): Z(i, l) at row i N + j
  let s = N * nb;
  for (l, zl) in zv.chunks_exact(nb).take(nl).enumerate() {
    for j in 0..N {
      let col = &mut tv[s * (j + N * l)..s * (j + N * l + 1)];
      for (y, &x) in col.chunks_exact_mut(N).zip(zl) {
        y[j] = x;
      }
    }
  }
}

/// C = op(A) op(B) (column-major, as dgemm with alpha 1, beta 0): from
/// GEMM_SMALL multiply-adds by the libxsmm kernel of the shape; below (the
/// call costs more than the product) or without a kernel by a plain loop
#[allow(clippy::too_many_arguments)]
fn gemm(
  ta: bool,
  tb: bool,
  (m, n, k): (usize, usize, usize),
  a: &[f64],
  lda: usize,
  b: &[f64],
  ldb: usize,
  c: &mut [f64],
  ldc: usize,
) {
  const GEMM_SMALL: usize = 512;
  if m * n * k >= GEMM_SMALL {
    // a tail call: the loop below keeps its registers
    return gemm_large((ta, tb), (m, n, k), a, lda, b, ldb, c, ldc);
  }
  gemm_loop((ta, tb), (m, n, k), a, lda, b, ldb, c, ldc);
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn gemm_large(
  (ta, tb): (bool, bool),
  (m, n, k): (usize, usize, usize),
  a: &[f64],
  lda: usize,
  b: &[f64],
  ldb: usize,
  c: &mut [f64],
  ldc: usize,
) {
  let s = GemmShape {
    ta,
    tb,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    accumulate: false,
  };
  if !kernel_gemm(&s, a, b, c) {
    gemm_loop((ta, tb), (m, n, k), a, lda, b, ldb, c, ldc);
  }
}

/// C = op(A) op(B) by a plain loop
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn gemm_loop(
  (ta, tb): (bool, bool),
  (m, n, k): (usize, usize, usize),
  a: &[f64],
  lda: usize,
  b: &[f64],
  ldb: usize,
  c: &mut [f64],
  ldc: usize,
) {
  // strides of op(A) (i, l) and op(B) (l, j)
  let (sai, sal) = if ta { (lda, 1) } else { (1, lda) };
  let (sbl, sbj) = if tb { (ldb, 1) } else { (1, ldb) };
  for j in 0..n {
    for i in 0..m {
      c[i + j * ldc] = (0..k)
        .map(|l| a[i * sai + l * sal] * b[l * sbl + j * sbj])
        .sum();
    }
  }
}

/// C = op(A) op(B), or C += op(A) op(B), by the libxsmm kernel of the
/// shape (found by libxsmm in its per-thread cache, about 10 ns); false
/// without one
#[inline(never)]
pub(crate) fn kernel_gemm(s: &GemmShape, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
  match Gemm::new(*s) {
    Some(g) => {
      g.call(a, b, c);
      true
    }
    None => false,
  }
}

/// Disjoint views of the buffer: the output o and the inputs a, b (tensors
/// never overlap the output of their instruction).
#[inline]
fn views(
  t: &mut [f64],
  o: Range<usize>,
  a: Range<usize>,
  b: Range<usize>,
) -> (&mut [f64], &[f64], &[f64]) {
  let (left, rest) = t.split_at_mut(o.start);
  let (mid, right) = rest.split_at_mut(o.len());
  let (left, right) = (&*left, &*right);
  let pick = |r: Range<usize>| -> &[f64] {
    if r.end <= o.start {
      &left[r]
    } else {
      &right[r.start - o.end..r.end - o.end]
    }
  };
  (mid, pick(a), pick(b))
}

/// GetFEM's fem_interpolation_context, geometry part: K = G grad(phi),
/// B, J at the points of the element
#[derive(Default)]
pub(crate) struct GeoCtx {
  /// mesh and reference dimensions
  pub n: usize,
  pub p: usize,
  pub g: Mat,
  pub k: Vec<f64>,
  pub b: Vec<f64>,
  pub j: f64,
}

impl GeoCtx {
  /// K, B, J at point ip of the precomputation
  pub fn at(&mut self, gp: &GeoPrecomp, ip: usize) -> Result<()> {
    let (n, p) = (self.n, self.p);
    if self.k.len() != n * p {
      self.k.resize(n * p, 0.0);
      self.b.resize(n * p, 0.0);
    }
    compute_K_matrix(
      self.g.data(),
      gp.grad[ip].data(),
      (n, self.g.ncols(), p),
      &mut self.k,
    );
    self.j = pseudo_inverse_small(&self.k, n, p, &mut self.b)
      .map_err(|_| Error::InvalidArgument("singular geometric transformation".into()))?;
    Ok(())
  }

  /// the point x = G phi(ip)
  pub fn x_at(&self, gp: &GeoPrecomp, ip: usize, x: &mut [f64]) {
    let n = self.n;
    x[..n].fill(0.0);
    for (l, a) in gp.val[ip].iter().enumerate() {
      for (r, xc) in x[..n].iter_mut().enumerate() {
        *xc += a * self.g[(r, l)];
      }
    }
  }

  /// the unit outward normal of face f (B n, n its reference normal,
  /// cleaned at 1e-13 as GetFEM) into normal; returns |B n|
  pub fn face_normal(&self, gt: &PGeoTrans, f: usize, normal: &mut [f64]) -> f64 {
    let nr = &gt.normals()[f];
    let (n, p) = (self.n, self.p);
    for (r, v) in normal[..n].iter_mut().enumerate() {
      *v = (0..p).map(|j| self.b[r + n * j] * nr[j]).sum();
    }
    let nup = normal[..n].iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in &mut normal[..n] {
      *v /= nup;
      if v.abs() < 1e-13 {
        *v = 0.0;
      }
    }
    nup
  }
}

/// An interpolate transformation as compiled: the target point is an
/// expression, the neighbor element, the contact runtime.
pub(crate) enum TransKind {
  Expr(Tree),
  Neighbor,
  Contact,
}

/// The kind of a transformation (see CompileCtx).
pub(crate) type TransFn<'a> = dyn Fn(&str) -> Option<TransKind> + 'a;
/// Its derivative w.r.t. a variable on the targets of a transformation.
pub(crate) type TransDerFn<'a> = dyn Fn(&str, &str, &str) -> Option<Option<Tree>> + 'a;

/// What the compilation reads: the variables, the fixed-size values.
pub(crate) struct CompileCtx<'a> {
  pub defs: &'a Defs,
  pub fixed: &'a dyn Fn(&str) -> Option<Vec<f64>>,
  /// inputs of an expression function (t, u)
  pub inputs: &'a [&'a str],
  /// qdim > 1 on scalar elements everywhere (GetFEM:
  /// is_uniformly_vectorized)
  pub vectorized: &'a dyn Fn(&str) -> bool,
  /// an interpolate transformation (None: not ported)
  pub trans: &'a TransFn<'a>,
  /// derivative of the target point of a transformation w.r.t. a variable
  /// (on the targets of a transformation), times its tests; Some(None): 0
  pub trans_der: &'a TransDerFn<'a>,
  /// an internal variable, condensed in this assembly (false everywhere:
  /// no condensation)
  pub internal: &'a dyn Fn(&str) -> bool,
}

/// The instruction set of an expression function (GetFEM: ga_function), None
/// if not ported (native functions, what reads the element).
fn compile_body(name: &str, cx: &CompileCtx) -> Option<GaFunction> {
  let u = crate::defs::function(name)?;
  if u.native.is_some() {
    return None;
  }
  let body = u.body.as_ref()?;
  let mut d = cx.defs.clone();
  let names: &[&str] = if u.nargs == 2 { &["t", "u"] } else { &["t"] };
  for nm in names {
    d.add(nm, true, Support::Fixed, &[1]);
  }
  let ctx = crate::semantic::Ctx {
    meshdim: 1,
    ref_elt_dim: 1,
    ..crate::semantic::Ctx::default()
  };
  let (tree, _) = crate::semantic::analyse_str(body, &d, &ctx).ok()?;
  let cx2 = CompileCtx {
    defs: &d,
    fixed: cx.fixed,
    inputs: names,
    vectorized: cx.vectorized,
    trans: cx.trans,
    trans_der: cx.trans_der,
    internal: cx.internal,
  };
  GaFunction::new(&tree, &cx2, &vec![1; names.len()])
}

/// the tensor of node i (instructions appended, children first, as
/// ga_compile_node), None if not ported
impl InstructionSet {
  fn sp(&self, s: u32) -> (u8, u8, u8) {
    self.sparsity.get(s as usize).copied().unwrap_or((0, 0, 0))
  }

  fn set_sp(&mut self, s: u32, v: (u8, u8, u8)) {
    let k = s as usize;
    if self.sparsity.len() <= k {
      self.sparsity.resize(k + 1, (0, 0, 0));
    }
    self.sparsity[k] = v;
  }

  /// GetFEM's ga_instruction_contraction_switch: the contraction t = a . b
  /// of size n by the sparsity of a, b
  fn contraction_switch(&mut self, t: u32, a: u32, b: u32, n: u32) -> Instruction {
    let (sa, sb) = (self.sp(a), self.sp(b));
    if sa.0 == 1 && sb.0 == 1 && sa.1 as u32 == n && sb.1 as u32 == n {
      self.set_sp(t, (10, sa.1, 0));
      return Instruction::ContractionOpt11 { t, a, b, nn: n };
    }
    if sa.0 == 2 && sb.0 == 2 && sa.1 == sb.1 && sa.2 == sb.2 && (sa.1 as u32) * (sa.2 as u32) == n
    {
      self.set_sp(t, (10, sa.1, 0));
      let nq = sa.2 as u32 | (sa.1 as u32) << 16;
      return Instruction::ContractionOpt22 { t, a, b, nq };
    }
    if sb.0 == 1 {
      return Instruction::ContractionOpt01 { t, a, b, nn: n };
    }
    if sb.0 == 2 && (sb.1 as u32) * (sb.2 as u32) == n {
      let nq = sb.2 as u32 | (sb.1 as u32) << 16;
      return Instruction::ContractionOpt02 { t, a, b, nq };
    }
    if sa.0 == 2 && (sa.1 as u32) * (sa.2 as u32) == n {
      let nq = sa.2 as u32 | (sa.1 as u32) << 16;
      return Instruction::ContractionOpt20 { t, a, b, nq };
    }
    Instruction::Contraction { t, a, b, i: n }
  }
}

/// A node without its children, the ids of its children: equal signatures,
/// equal subtrees (GetFEM compares subtrees with the same hash,
/// sub_tree_are_equal)
#[derive(Clone, PartialEq, Eq, Hash)]
struct Sig {
  ty: Ty,
  names: [String; 9],
  test: usize,
  der: (usize, usize, usize),
  sizes: Vec<usize>,
  data: Vec<u64>,
  children: Vec<u32>,
}

impl InstructionSet {
  /// the ids of the subtrees of t (children first, one pass)
  fn subtree_ids(&mut self, t: &Tree) -> Vec<u32> {
    let mut ids = vec![NONE; t.n.len()];
    fn visit(t: &Tree, i: usize, gis: &mut InstructionSet, ids: &mut [u32]) {
      let n = &t.n[i];
      for &c in &n.children {
        visit(t, c, gis, ids);
      }
      let sig = Sig {
        ty: n.ty,
        names: [
          n.name.clone(),
          n.name_test1.clone(),
          n.name_test2.clone(),
          n.interp_test1.clone(),
          n.interp_test2.clone(),
          n.interpolate_name.clone(),
          n.interpolate_name_der.clone(),
          n.elementary_name.clone(),
          n.elementary_target.clone(),
        ],
        test: n.test,
        der: (n.der1, n.der2, n.nbc1),
        sizes: n.sizes.clone(),
        data: if matches!(n.ty, Ty::Constant) {
          n.data.iter().map(|x| x.to_bits()).collect()
        } else {
          vec![]
        },
        children: n.children.iter().map(|&c| ids[c]).collect(),
      };
      let next = gis.sigs.len() as u32;
      ids[i] = *gis.sigs.entry(sig).or_insert(next);
    }
    if let Some(r) = t.root {
      visit(t, r, self, &mut ids);
    }
    ids
  }
}

fn node(t: &Tree, i: usize, gis: &mut InstructionSet, cx: &CompileCtx) -> Option<TensorId> {
  // the subtree ids of t, computed on its first node, dropped with it
  let ptr = t as *const Tree as usize;
  let own = !gis.trees.iter().any(|(q, _)| *q == ptr);
  if own {
    let ids = gis.subtree_ids(t);
    gis.trees.push((ptr, ids));
  }
  let k = gis.trees.iter().rev().find(|(q, _)| *q == ptr).unwrap().1[i];
  let s = match gis.node_list.get(&k) {
    Some(&s) => Some(s),
    None => {
      let s = node_new(t, i, gis, cx);
      if let Some(s) = s {
        gis.node_list.insert(k, s);
      }
      s
    }
  };
  if own {
    gis.trees.pop();
  }
  s
}

fn node_new(t: &Tree, i: usize, gis: &mut InstructionSet, cx: &CompileCtx) -> Option<TensorId> {
  let n = &t.n[i];
  let size = n.data.len();
  let ch = n.children.clone();
  let ins = |gis: &mut InstructionSet, x: Instruction| gis.instructions.push(x);
  // the result tensor: sized by the test functions if any
  let out = |gis: &mut InstructionSet| -> Option<u32> {
    if n.test == 0 {
      return Some(gis.zeros(size));
    }
    let tv = |gis: &mut InstructionSet, name: &str, tr: &str| -> Option<u32> {
      if name.is_empty() {
        return Some(NONE);
      }
      match cx.defs.var(name)?.support {
        Support::Fem { .. } if !tr.is_empty() => Some(gis.var(&interp_key(name, tr))),
        Support::Fem { .. } | Support::Fixed | Support::ImData if tr.is_empty() => {
          Some(gis.var(name))
        }
        _ => None,
      }
    };
    let (t1, t2) = (
      tv(gis, &n.name_test1, &n.interp_test1)?,
      tv(gis, &n.name_test2, &n.interp_test2)?,
    );
    Some(gis.shaped(Shape::Test {
      proper: n.tensor_proper_size() as u32,
      t1,
      t2,
    }))
  };
  Some(match n.ty {
    Ty::InterpEltK | Ty::InterpEltB if n.test == 0 => {
      let tr = gis.tr(&n.interpolate_name)?;
      let s = gis.zeros(size);
      let b = (n.ty == Ty::InterpEltB) as u32;
      ins(gis, Instruction::InterpElt { t: s, tr, b });
      s
    }
    Ty::InterpX | Ty::InterpNormal if n.test == 0 => {
      let tr = gis.tr(&n.interpolate_name)?;
      let s = gis.zeros(size);
      ins(
        gis,
        if n.ty == Ty::InterpX {
          Instruction::InterpX { t: s, tr }
        } else {
          Instruction::InterpNormal { t: s, tr }
        },
      );
      s
    }
    Ty::Field {
      src: Src::Interp,
      der,
      test,
    } => {
      if !matches!(cx.defs.var(&n.name)?.support, Support::Fem { .. }) {
        return None;
      }
      let _ = gis.tr(&n.interpolate_name)?;
      let v = gis.var(&interp_key(&n.name, &n.interpolate_name));
      let s = if test { out(gis)? } else { gis.zeros(size) };
      let der = match der {
        Der::Val => 0,
        Der::Grad => 1,
        Der::Hess => 2,
        Der::Div => 3,
      };
      if der == 2 {
        gis.needs_hess = true;
      }
      ins(
        gis,
        Instruction::InterpField {
          t: s,
          v,
          der,
          test: test as u8,
        },
      );
      s
    }
    // the derivative of the target point, compiled at the source point
    Ty::InterpolateDerivative => {
      let tr = gis.tr(&n.interpolate_name_der)?;
      if gis.contact[tr as usize] {
        gis.trans_ders[tr as usize] = true;
        let key = if n.interpolate_name.is_empty() {
          n.name.clone()
        } else {
          interp_key(&n.name, &n.interpolate_name)
        };
        let v = gis.var(&key);
        let s = out(gis)?;
        ins(gis, Instruction::TransDer { t: s, tr, v });
        return Some(s);
      }
      match (cx.trans_der)(&n.interpolate_name_der, &n.name, &n.interpolate_name)? {
        Some(d) => {
          let r = d.root?;
          if d.n[r].sizes != n.sizes {
            return None;
          }
          node(&d, r, gis, cx)?
        }
        None => out(gis)?,
      }
    }
    Ty::InterpolateFilter => {
      let tr = gis.tr_filter(&n.interpolate_name)?;
      let s = out(gis)?;
      ins(
        gis,
        Instruction::Filter {
          t: s,
          tr,
          st: if n.nbc1 == usize::MAX {
            NONE
          } else {
            n.nbc1 as u32
          },
          len: 0,
        },
      );
      let at = gis.instructions.len();
      // nothing computed inside is reused outside (it may be skipped)
      let saved = (
        gis.node_list.clone(),
        gis.tests_copied.clone(),
        gis.base_val.clone(),
        gis.base_grad.clone(),
        gis.base_hess.clone(),
        gis.xfem_base.clone(),
      );
      gis.in_filter += 1;
      let c = node(t, n.children[0], gis, cx);
      gis.in_filter -= 1;
      (
        gis.node_list,
        gis.tests_copied,
        gis.base_val,
        gis.base_grad,
        gis.base_hess,
        gis.xfem_base,
      ) = saved;
      // variables first met inside: not computed outside
      let nv = gis.vars.len();
      for b in [&mut gis.base_val, &mut gis.base_grad, &mut gis.base_hess] {
        b.resize(nv, NONE);
      }
      let c = c?;
      ins(gis, Instruction::Copy { t: s, a: c });
      let len = (gis.instructions.len() - at) as u32;
      if let Instruction::Filter { len: l, .. } = &mut gis.instructions[at - 1] {
        *l = len;
      }
      s
    }
    Ty::Field {
      src: Src::Elem,
      der,
      test,
    } => {
      let (v, dest) = (&n.name, &n.elementary_target);
      for w in [v, dest] {
        if !matches!(cx.defs.var(w)?.support, Support::Fem { .. }) {
          return None;
        }
      }
      let (vs, vd) = (gis.var(v), gis.var(dest));
      let key = (n.elementary_name.clone(), vs, vd);
      let m = match gis.elts.iter().position(|e| *e == key) {
        Some(e) => gis.elt_mats[e],
        None => {
          let m = gis.shaped(Shape::ElemMat(vd, vs));
          gis.elts.push(key);
          gis.elt_mats.push(m);
          let e = (gis.elts.len() - 1) as u32;
          gis.elt_instructions.push(Instruction::EltTrans { m, e });
          m
        }
      };
      if test {
        // the tests of dest mapped onto the DOFs of v
        let sd = gis.shaped(Shape::Test {
          proper: n.tensor_proper_size() as u32,
          t1: vd,
          t2: NONE,
        });
        gis.copy_test(cx, vd, der, 0, sd, n3(&n.sizes))?;
        let s = out(gis)?;
        ins(
          gis,
          Instruction::ElemTest {
            t: s,
            a: sd,
            m,
            d: vd,
          },
        );
        s
      } else {
        let c = gis.coef_of(vs);
        let co = gis.shaped(Shape::Coef(vd));
        gis
          .elt_instructions
          .push(Instruction::ElemCoef { t: co, m, c });
        let s = gis.zeros(size);
        gis.field_val(vd, co, der, 0, s);
        s
      }
    }
    Ty::Constant if n.test == 0 => gis.const_tensor(&n.data),
    Ty::Zero if n.test == 0 => gis.zeros(size),
    Ty::EltSize => {
      gis.needs_elt_size = true;
      let s = gis.zeros(size);
      ins(gis, Instruction::EltSize { t: s });
      s
    }
    Ty::EltK | Ty::EltB => {
      gis.needs_kb = true;
      let s = gis.zeros(size);
      ins(
        gis,
        if n.ty == Ty::EltK {
          Instruction::EltK { t: s }
        } else {
          Instruction::EltB { t: s }
        },
      );
      s
    }
    Ty::CMatrix => {
      let mut comps = vec![];
      for &c in &ch {
        comps.push(node(t, c, gis, cx)?);
      }
      if n.test == 0 && comps.iter().any(|&c| gis.shapes.get(c as usize).is_none()) {
        return None;
      }
      gis.maps.push(comps);
      let l = (gis.maps.len() - 1) as u32;
      let s = out(gis)?;
      ins(
        gis,
        if n.test == 0 {
          Instruction::CMatrix { t: s, l }
        } else {
          Instruction::CMatrixTests { t: s, l }
        },
      );
      s
    }
    Ty::Normal => {
      gis.needs_normal = true;
      let s = gis.zeros(size);
      ins(gis, Instruction::Normal { t: s });
      s
    }
    Ty::Field {
      src: src @ (Src::Plain | Src::XfemPlus | Src::XfemMinus),
      der,
      test: true,
    } => {
      let side = side_of(src);
      // fixed-size and im variables: the identity set by the semantic
      // analysis (GetFEM: no instruction)
      if matches!(
        cx.defs.var(&n.name)?.support,
        Support::Fixed | Support::ImData
      ) && der == Der::Val
      {
        return Some(gis.const_tensor(&n.data));
      }
      if !matches!(cx.defs.var(&n.name)?.support, Support::Fem { .. }) {
        return None;
      }
      let vi = gis.var(&n.name);
      let s = out(gis)?;
      match gis.tests_copied.get(&(vi, der, side)) {
        // the other test function of the variable: its copy's storage
        Some(&s0) => {
          gis.alias.insert(s, s0);
          let sp = gis.sp(s0);
          gis.set_sp(s, sp);
        }
        None => {
          gis.copy_test(cx, vi, der, side, s, n3(&n.sizes))?;
          gis.tests_copied.insert((vi, der, side), s);
        }
      }
      s
    }
    Ty::X => {
      gis.needs_x = true;
      let s = gis.zeros(size);
      if n.nbc1 > 0 {
        ins(
          gis,
          Instruction::XComp {
            t: s,
            k: (n.nbc1 - 1) as u32,
          },
        );
      } else {
        ins(gis, Instruction::X { t: s });
      }
      s
    }
    Ty::Field {
      src: src @ (Src::Plain | Src::XfemPlus | Src::XfemMinus),
      der,
      test: false,
    } => {
      let side = side_of(src);
      if let Some(k) = cx.inputs.iter().position(|x| *x == n.name) {
        return Some(gis.inputs[k]);
      }
      let v = cx.defs.var(&n.name)?;
      match v.support {
        Support::Fixed => {
          let val = (cx.fixed)(&n.name)?;
          if val.len() != size {
            return None;
          }
          gis.const_tensor(&val)
        }
        Support::ImData if der == Der::Val => {
          let vi = gis.var(&n.name);
          let s = gis.zeros(size);
          ins(gis, Instruction::ImVal { t: s, v: vi });
          s
        }
        Support::Fem { .. } => {
          let vi = gis.var(&n.name);
          let c = gis.coef_of(vi);
          let s = gis.zeros(size);
          gis.field_val(vi, c, der, side, s);
          s
        }
        _ => return None,
      }
    }
    Ty::Op(op) => {
      let a = node(t, ch[0], gis, cx)?;
      let c0 = &t.n[ch[0]];
      let sa = c0.data.len();
      match op {
        Tok::UnaryMinus => {
          let s = out(gis)?;
          let m1 = gis.const_tensor(&[-1.0]);
          ins(gis, Instruction::ScalarMult { t: s, a, c: m1 });
          let sp = gis.sp(a);
          gis.set_sp(s, sp);
          s
        }
        // a shallow copy for a scalar (GetFEM: set_to_copy)
        Tok::Quote if n.tensor_proper_size() == 1 => a,
        Tok::Sym if n.tensor_proper_size() == 1 => a,
        Tok::Sym | Tok::Skew => {
          let ord = c0.sizes.len();
          let (s1, s2) = (c0.sizes[ord - 2], c0.sizes[ord - 1]);
          let s = out(gis)?;
          let ss = (s1 | (s2 << 16)) as u32;
          ins(
            gis,
            if op == Tok::Sym {
              Instruction::Sym { t: s, a, s: ss }
            } else {
              Instruction::Skew { t: s, a, s: ss }
            },
          );
          s
        }
        Tok::Deviator => {
          let nn = if c0.tensor_proper_size() == 1 {
            1
          } else {
            *c0.sizes.last()?
          };
          let s = out(gis)?;
          ins(
            gis,
            Instruction::Deviator {
              t: s,
              a,
              n: nn as u32,
            },
          );
          s
        }
        // GetFEM: a copy for a trace of size 1
        Tok::Trace => {
          let nn = if c0.tensor_proper_size() == 1 {
            1
          } else {
            *c0.sizes.last()?
          };
          if nn == 1 {
            a
          } else {
            let s = out(gis)?;
            ins(
              gis,
              Instruction::Trace {
                t: s,
                a,
                n: nn as u32,
              },
            );
            s
          }
        }
        Tok::Quote => {
          let n1 = c0.proper_size(0);
          let n2 = if c0.tensor_order() > 1 {
            c0.proper_size(1)
          } else {
            1
          };
          let nn: usize = (2..c0.tensor_order()).map(|k| c0.proper_size(k)).product();
          let s = out(gis)?;
          ins(
            gis,
            Instruction::Transpose {
              t: s,
              a,
              jk: (n1 | (n2 << 16)) as u32,
              i: nn as u32,
            },
          );
          s
        }
        Tok::Plus
        | Tok::Minus
        | Tok::Mult
        | Tok::Dot
        | Tok::Colon
        | Tok::Div
        | Tok::TMult
        | Tok::DotMult
        | Tok::DotDiv => {
          let b = node(t, ch[1], gis, cx)?;
          let c1 = &t.n[ch[1]];
          // a scalar without test functions, as GetFEM's tensor().size()
          let (sa, sb) = (
            if c0.test == 0 { sa } else { usize::MAX },
            if c1.test == 0 {
              c1.data.len()
            } else {
              usize::MAX
            },
          );
          let (dim0, dim1) = (c0.tensor_order(), c1.tensor_order());
          let (tps0, tps1) = (c0.tensor_proper_size(), c1.tensor_proper_size());
          let s2 = ((tps0 * tps1 / n.tensor_proper_size()) as f64)
            .sqrt()
            .round() as u32;
          let s = out(gis)?;
          // GetFEM's choice of instruction (ga_compile_node, GA_DOT,
          // GA_COLON, GA_MULT, GA_TMULT)
          let tt = n.test;
          let pack2 = |x: usize, y: usize| (x | (y << 16)) as u32;
          let pack3 = |j: usize, i: usize, k: usize| (j | (i << 10) | (k << 20)) as u32;
          let (su, j2) = (s2 as usize, s2);
          let x = match op {
            Tok::Plus => Instruction::Add { t: s, a, b },
            Tok::Minus => Instruction::Sub { t: s, a, b },
            Tok::Div | Tok::DotDiv if sb == 1 => Instruction::ScalarDiv { t: s, a, c: b },
            Tok::Div => return None,
            Tok::DotDiv if c1.test == 0 => Instruction::DotDiv { t: s, a, b },
            Tok::DotDiv => return None,
            _ if sa == 1 => Instruction::ScalarMult { t: s, a: b, c: a },
            _ if sb == 1 => Instruction::ScalarMult { t: s, a, c: b },
            Tok::DotMult if c1.test == 0 => Instruction::DotMult { t: s, a, b },
            Tok::DotMult if c0.test == 0 => Instruction::DotMult { t: s, a: b, b: a },
            Tok::DotMult if c0.test == 1 => Instruction::DotMultSpec { t: s, a, b },
            Tok::DotMult => Instruction::DotMultSpec { t: s, a: b, b: a },
            Tok::TMult if c1.test == 0 => Instruction::SimpleTmult { t: s, a, b },
            // spec(2)_tmult put the tests of child0 (child1) first: test1
            // must come first (bit 31: test2 of child0 after test1)
            Tok::TMult if c0.test == 2 && c1.test == 1 => Instruction::SpecTmult {
              t: s,
              a,
              b,
              ij: pack2(tps0, tps1) | 1 << 31,
            },
            Tok::TMult if c0.test == 1 && c1.test == 2 => Instruction::SpecTmult {
              t: s,
              a,
              b,
              ij: pack2(tps0, tps1),
            },
            Tok::TMult if tps1 == 1 => Instruction::Spec2Tmult { t: s, a, b },
            Tok::TMult => Instruction::SpecTmult {
              t: s,
              a,
              b,
              ij: pack2(tps0, tps1),
            },
            _ if (op == Tok::Dot && dim1 <= 1)
              || (op == Tok::Colon && dim1 <= 2)
              || (op == Tok::Mult && (dim0 == 4 || dim1 <= 1))
              || tps1 == 1 =>
            {
              if tt < 3 {
                if tps0 == 1 {
                  Instruction::SimpleTmult { t: s, a, b }
                } else if tps1 == 1 {
                  Instruction::SimpleTmult { t: s, a: b, b: a }
                } else {
                  Instruction::Contraction { t: s, a, b, i: j2 }
                }
              } else if c1.test == 1 || c1.test == 3 {
                if c1.test == 3 || tps1 <= su {
                  if tps0 == 1 {
                    Instruction::SimpleTmult { t: s, a: b, b: a }
                  } else {
                    Instruction::Contraction { t: s, a, b, i: j2 }
                  }
                } else {
                  Instruction::SpecContraction {
                    t: s,
                    a: b,
                    b: a,
                    nn: j2,
                  }
                }
              } else if c1.test == 0 || (tps0 == su && tps1 == su) {
                if tps0 == 1 {
                  Instruction::SimpleTmult { t: s, a, b }
                } else {
                  Instruction::Contraction {
                    t: s,
                    a: b,
                    b: a,
                    i: j2,
                  }
                }
              } else if tps0 == su {
                Instruction::Contraction {
                  t: s,
                  a: b,
                  b: a,
                  i: j2,
                }
              } else if tps1 == su {
                Instruction::SpecContraction { t: s, a, b, nn: j2 }
              } else {
                Instruction::Spec2Contraction { t: s, a, b, nn: j2 }
              }
            }
            _ if tt < 3 => {
              if tps0 == 1 {
                Instruction::SimpleTmult { t: s, a, b }
              } else if c1.test == 0 {
                Instruction::MatrixMult { t: s, a, b, j: j2 }
              } else {
                Instruction::MatrixMultSpec {
                  t: s,
                  a,
                  b,
                  jik: pack3(su, tps0 / su, tps1 / su),
                }
              }
            }
            _ => {
              if tps0 == 1 {
                if c0.test == 0 || c0.test == 1 {
                  Instruction::SimpleTmult { t: s, a, b }
                } else {
                  Instruction::SpecTmult {
                    t: s,
                    a: b,
                    b: a,
                    ij: pack2(tps1, tps0),
                  }
                }
              } else if c1.test == 0 {
                Instruction::MatrixMult { t: s, a, b, j: j2 }
              } else if c1.test == 2 {
                Instruction::MatrixMultSpec {
                  t: s,
                  a,
                  b,
                  jik: pack3(su, tps0 / su, tps1 / su),
                }
              } else {
                Instruction::MatrixMultSpec2 {
                  t: s,
                  a,
                  b,
                  jik: pack3(su, tps0 / su, tps1 / su),
                }
              }
            }
          };
          // sparsity: kept by scalar factors and sums (GetFEM), chosen by
          // the contractions
          let x = match x {
            Instruction::Contraction { t, a, b, i } => gis.contraction_switch(t, a, b, i),
            Instruction::ScalarMult { a, .. } | Instruction::ScalarDiv { a, .. } => {
              let sp = gis.sp(a);
              gis.set_sp(s, sp);
              x
            }
            Instruction::Add { a, b, .. } | Instruction::Sub { a, b, .. } => {
              if gis.sp(a) == gis.sp(b) {
                let sp = gis.sp(a);
                gis.set_sp(s, sp);
              }
              x
            }
            _ => x,
          };
          ins(gis, x);
          s
        }
        _ => return None,
      }
    }
    Ty::Params => {
      let f = &t.n[ch[0]];
      match f.ty {
        Ty::PredefFunc => {
          if f.der1 != 0 || ch.len() > 3 {
            return None;
          }
          let a = node(t, ch[1], gis, cx)?;
          let b = if ch.len() == 3 {
            node(t, ch[2], gis, cx)?
          } else {
            NONE
          };
          let s = gis.zeros(size);
          let x = match crate::names::function(&f.name).map(|x| x.eval) {
            Some(crate::names::Eval::One(g)) if b == NONE => {
              gis.funcs.push(g);
              Instruction::Func1 {
                t: s,
                a,
                f: (gis.funcs.len() - 1) as u32,
              }
            }
            Some(crate::names::Eval::Two(g)) if b != NONE => {
              gis.funcs2.push(g);
              Instruction::Func2 {
                t: s,
                a,
                b,
                f: (gis.funcs2.len() - 1) as u32,
              }
            }
            Some(_) => return None,
            None
              if crate::defs::function(&f.name)
                .is_some_and(|u| u.native.is_some() && u.nargs == ch.len() - 1) =>
            {
              let g = crate::defs::function(&f.name).unwrap().native.unwrap();
              gis.natives.push(g);
              Instruction::FuncNative {
                t: s,
                a,
                b,
                f: (gis.natives.len() - 1) as u32,
              }
            }
            None => {
              let sub = compile_body(&f.name, cx)?;
              if sub.gis.inputs.len() != ch.len() - 1 {
                return None;
              }
              gis.subs.push(sub);
              Instruction::FuncExpr {
                t: s,
                a,
                b,
                f: (gis.subs.len() - 1) as u32,
              }
            }
          };
          ins(gis, x);
          s
        }
        // the same values (GetFEM: copy_tensor); shared when no sparsity
        Ty::Reshape => {
          let a = node(t, ch[1], gis, cx)?;
          if gis.sp(a).0 == 0 {
            a
          } else {
            let s = out(gis)?;
            ins(gis, Instruction::Copy { t: s, a });
            s
          }
        }
        Ty::Operator if ch.len() <= 3 => {
          let op = cx.defs.operators.get(&f.name)?.clone();
          let a = node(t, ch[1], gis, cx)?;
          let b = if ch.len() == 3 {
            node(t, ch[2], gis, cx)?
          } else {
            NONE
          };
          let s = gis.zeros(size);
          let shapes = ch[1..].iter().map(|&c| t.n[c].sizes.clone()).collect();
          gis.nonlins.push(NlCall {
            op,
            der: (f.der1, f.der2),
            shapes,
          });
          let k = (gis.nonlins.len() - 1) as u32;
          ins(gis, Instruction::Nonlin { t: s, a, b, op: k });
          s
        }
        Ty::CrossProduct if ch.len() == 3 => {
          let a = node(t, ch[1], gis, cx)?;
          let b = node(t, ch[2], gis, cx)?;
          let (c1, c2) = (&t.n[ch[1]], &t.n[ch[2]]);
          let s = out(gis)?;
          let x = if c1.test == 2 && c2.test == 1 {
            Instruction::CrossTf {
              t: s,
              a: b,
              b: a,
              inv: 1,
            }
          } else if c1.test != 0 || c2.test != 0 {
            Instruction::CrossTf { t: s, a, b, inv: 0 }
          } else {
            Instruction::Cross { t: s, a, b }
          };
          ins(gis, x);
          s
        }
        Ty::IndMoveLast if ch.len() == 3 => {
          let a = node(t, ch[1], gis, cx)?;
          let c1 = &t.n[ch[1]];
          let ind = (t.n[ch[2]].data[0].round() as usize).checked_sub(1)?;
          let ii2: usize = (ind + 1..c1.tensor_order())
            .map(|k| c1.proper_size(k))
            .product();
          let nn = c1.proper_size(ind);
          gis.maps.push(vec![nn as u32, ii2 as u32]);
          let l = (gis.maps.len() - 1) as u32;
          let s = out(gis)?;
          ins(gis, Instruction::IndMoveLast { t: s, a, l });
          s
        }
        Ty::SwapInd if ch.len() == 4 => {
          let a = node(t, ch[1], gis, cx)?;
          let c1 = &t.n[ch[1]];
          let mut i2 = (t.n[ch[2]].data[0].round() as usize).checked_sub(1)?;
          let mut i3 = (t.n[ch[3]].data[0].round() as usize).checked_sub(1)?;
          if i2 > i3 {
            std::mem::swap(&mut i2, &mut i3);
          }
          let (mut ii2, mut ii3) = (1, 1);
          for k in 0..c1.tensor_order() {
            if k > i2 && k < i3 {
              ii2 *= c1.proper_size(k);
            }
            if k > i3 {
              ii3 *= c1.proper_size(k);
            }
          }
          let (nn1, nn2) = (c1.proper_size(i2), c1.proper_size(i3));
          gis
            .maps
            .push(vec![nn1 as u32, nn2 as u32, ii2 as u32, ii3 as u32]);
          let l = (gis.maps.len() - 1) as u32;
          let s = out(gis)?;
          ins(gis, Instruction::SwapInd { t: s, a, l });
          s
        }
        Ty::Contract if ch.len() == 4 || ch.len() == 5 => {
          let idx = |k: usize| (t.n[ch[k]].data[0].round() as usize).checked_sub(1);
          let a = node(t, ch[1], gis, cx)?;
          let c1 = &t.n[ch[1]];
          if ch.len() == 4 {
            let (mut i1, mut i2) = (idx(2)?, idx(3)?);
            if i1 > i2 {
              std::mem::swap(&mut i1, &mut i2);
            }
            let nn = c1.proper_size(i1);
            let (mut ii2, mut ii3) = (1, 1);
            for k in 0..c1.tensor_order() {
              if k > i1 && k < i2 {
                ii2 *= c1.proper_size(k);
              }
              if k > i2 {
                ii3 *= c1.proper_size(k);
              }
            }
            gis.maps.push(vec![nn as u32, ii2 as u32, ii3 as u32]);
            let l = (gis.maps.len() - 1) as u32;
            let s = out(gis)?;
            ins(gis, Instruction::Contract11 { t: s, a, l });
            s
          } else {
            let b = node(t, ch[3], gis, cx)?;
            let c2 = &t.n[ch[3]];
            // test1, test2 order: contract_2_1_rev, not yet
            if c1.test == 1 && c2.test == 2 {
              return None;
            }
            let (i1, i2) = (idx(2)?, idx(4)?);
            let nn = c1.proper_size(i1);
            let (mut ii1, mut ii2, mut ii3, mut ii4) = (1, 1, 1, 1);
            for k in 0..c1.tensor_order() {
              if k < i1 {
                ii1 *= c1.proper_size(k);
              }
              if k > i1 {
                ii2 *= c1.proper_size(k);
              }
            }
            for k in 0..c2.tensor_order() {
              if k < i2 {
                ii3 *= c2.proper_size(k);
              }
              if k > i2 {
                ii4 *= c2.proper_size(k);
              }
            }
            gis.maps.push(vec![
              nn as u32, ii1 as u32, ii2 as u32, ii3 as u32, ii4 as u32,
            ]);
            let l = (gis.maps.len() - 1) as u32;
            let s = out(gis)?;
            ins(gis, Instruction::Contract21 { t: s, a, b, l });
            s
          }
        }
        // components of a tensor (GetFEM: tensor_slice), test indices kept
        Ty::Constant
        | Ty::Zero
        | Ty::Field { .. }
        | Ty::Op(_)
        | Ty::Params
        | Ty::X
        | Ty::Normal
        | Ty::EltK
        | Ty::EltB
        | Ty::InterpX
        | Ty::InterpNormal
        | Ty::InterpEltK
        | Ty::InterpEltB
        | Ty::InterpolateFilter
        | Ty::InterpolateDerivative
        | Ty::CMatrix => {
          let a = node(t, ch[0], gis, cx)?;
          let c0 = &t.n[ch[0]];
          let ord = c0.tensor_order();
          if ch.len() != ord + 1 {
            return None;
          }
          let sz: Vec<usize> = (0..ord).map(|k| c0.proper_size(k)).collect();
          let mut fixed = vec![0usize; ord];
          let mut free = vec![];
          for (k, &c) in ch[1..].iter().enumerate() {
            let cn = &t.n[c];
            match cn.ty {
              Ty::AllIndices => free.push(k),
              Ty::Constant => fixed[k] = (cn.data[0].round() as usize).checked_sub(1)?,
              _ => return None,
            }
          }
          let nfree: usize = free.iter().map(|&k| sz[k]).product();
          let mut map = Vec::with_capacity(nfree);
          for rr in 0..nfree {
            let mut idx = fixed.clone();
            let mut q = rr;
            for &k in &free {
              idx[k] = q % sz[k];
              q /= sz[k];
            }
            let mut off = 0;
            for k in (0..ord).rev() {
              off = off * sz[k] + idx[k];
            }
            map.push(off as u32);
          }
          gis.maps.push(map);
          let s = out(gis)?;
          let mi = (gis.maps.len() - 1) as u32;
          ins(gis, Instruction::Gather { t: s, a, map: mi });
          s
        }
        _ => return None,
      }
    }
    _ => return None,
  })
}

/// A variable on the target elements of a transformation (its own tests).
pub(crate) fn interp_key(var: &str, tr: &str) -> String {
  format!("{var}\u{1}{tr}")
}

/// An expression of fixed-size inputs evaluated at a point, without fields
/// (GetFEM ga_function: functions given by an expression, obstacle level
/// sets, global functions). Evaluated by eval, or within an instruction set
/// in a state of its own per thread (ExecState::subs).
pub(crate) struct GaFunction {
  pub gis: InstructionSet,
  pub out: u32,
  /// the sizes of the tests of the inputs (derivatives)
  dims: Vec<VarDims>,
  m: std::sync::Mutex<ExecState>,
}

impl Clone for GaFunction {
  fn clone(&self) -> Self {
    GaFunction {
      gis: self.gis.clone(),
      out: self.out,
      dims: self.dims.clone(),
      m: std::sync::Mutex::new(self.exec_state()),
    }
  }
}

/// the side of a field on a level set: 0, or 1, -1 (Xfem_plus, Xfem_minus)
fn side_of(src: Src) -> i8 {
  match src {
    Src::XfemPlus => 1,
    Src::XfemMinus => -1,
    _ => 0,
  }
}

/// Instructions without fields or elements.
struct NoExt;

impl Ext for NoExt {
  fn slice_dofs(&mut self, _: usize, _: &mut [f64]) {}
  fn val_base(&mut self, _: usize, _: &mut [f64]) {}
  fn grad_base(&mut self, _: usize, _: &mut [f64]) {}
  fn hess_base(&mut self, _: usize, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn xfem_base(&mut self, _: usize, _: u8, _: i8, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn im_val(&mut self, _: usize, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn trans_call(&mut self, _: usize, _: &[f64], _: &[f64], _: bool, _: bool) -> Result<()> {
    Ok(())
  }
  fn trans_der(&mut self, _: usize, _: usize, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn trans_status(&self, _: usize) -> usize {
    0
  }
  fn interp_x(&self, _: usize, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn interp_normal(&self, _: usize, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn interp_elt(&self, _: usize, _: bool, _: &mut [f64]) {}
  fn interp_field(&mut self, _: usize, _: u8, _: bool, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn elem_trans(&mut self, _: usize, _: &mut [f64]) -> Result<()> {
    Ok(())
  }
  fn point_value(&mut self, _: &[f64]) -> Result<()> {
    Ok(())
  }
  fn vector(&mut self, _: usize, _: &[f64]) {}
  fn matrix(&mut self, _: usize, _: usize, _: &[f64]) {}
  fn matrix_opt10(&mut self, _: usize, _: usize, _: usize, _: &[f64]) {}
}

impl GaFunction {
  /// tree of the inputs cx.inputs (sizes), their tests only as fields
  pub(crate) fn new(tree: &Tree, cx: &CompileCtx, sizes: &[usize]) -> Option<GaFunction> {
    let mut gis = InstructionSet::default();
    for &n in sizes {
      let s = gis.zeros(n);
      gis.inputs.push(s);
    }
    let out = node(tree, tree.root?, &mut gis, cx)?;
    if gis.needs_x
      || gis.needs_normal
      || gis.needs_elt_size
      || gis.needs_kb
      || !gis.trans.is_empty()
      || !gis.elt_instructions.is_empty()
    {
      return None;
    }
    let mut dims = vec![];
    for v in &gis.vars {
      // the tests of an input (derivatives)
      let i = cx.inputs.iter().position(|x| x == v)?;
      dims.push(VarDims {
        nb: sizes[i],
        td: 1,
        qdim: 1,
        n: sizes[i],
      });
    }
    let mut f = GaFunction {
      gis,
      out,
      dims,
      m: Default::default(),
    };
    f.m = std::sync::Mutex::new(f.exec_state());
    Some(f)
  }

  /// a new execution state
  pub(crate) fn exec_state(&self) -> ExecState {
    let mut m = self.gis.exec_state();
    m.dims.clone_from(&self.dims);
    m.layout(&self.gis);
    m
  }

  /// the value at the inputs
  pub(crate) fn eval(&self, inputs: &[&[f64]], out: &mut Vec<f64>) -> Result<()> {
    let mut m = self.m.lock().unwrap();
    for (&s, x) in self.gis.inputs.iter().zip(inputs) {
      let r = m.range(s);
      m.buffer[r].copy_from_slice(x);
    }
    self.gis.run(&self.gis.instructions, &mut m, &mut NoExt)?;
    out.clear();
    out.extend_from_slice(&m.buffer[m.range(self.out)]);
    Ok(())
  }
}

/// The values of an expression at the points (GetFEM ga_interpolation), if
/// all is ported.
pub(crate) fn compile_values(trees: &[Tree], cx: &CompileCtx) -> Option<InstructionSet> {
  let mut gis = InstructionSet::default();
  gis.transformations(trees, cx)?;
  let [t] = trees else {
    return None;
  };
  let r = t.root?;
  if t.n[r].test != 0 {
    return None;
  }
  let a = node(t, r, &mut gis, cx)?;
  gis.instructions.push(Instruction::PointValue { a });
  gis.values = true;
  Some(gis)
}

/// A scalar assembly (order 0) of analysed trees, if all is ported.
pub(crate) fn compile_scalar(trees: &[Tree], cx: &CompileCtx) -> Option<InstructionSet> {
  let rw = rewrite::all(trees);
  let trees = rw.as_deref().unwrap_or(trees);
  let mut gis = InstructionSet::default();
  gis.transformations(trees, cx)?;
  for t in trees {
    let r = t.root?;
    if t.n[r].sizes.iter().product::<usize>() != 1 {
      return None;
    }
    let a = node(t, r, &mut gis, cx)?;
    gis.instructions.push(Instruction::AddToCoeff { a });
  }
  gis.scalar = true;
  Some(gis)
}

/// A matrix assembly (order 2), if all is ported.
pub(crate) fn compile_matrix(trees: &[Tree], cx: &CompileCtx) -> Option<InstructionSet> {
  let rw = rewrite::all(trees);
  let trees = rw.as_deref().unwrap_or(trees);
  #[cfg(test)]
  TREES.with(|c| c.borrow_mut().extend(trees.iter().map(Tree::debug_string)));
  let mut gis = InstructionSet::default();
  gis.transformations(trees, cx)?;
  let mut blocks = CondensationBlocks::default();
  for t in trees {
    let r = t.root?;
    let n = &t.n[r];
    if n.test != 3 || n.tensor_proper_size() != 1 {
      return None;
    }
    let a = node(t, r, &mut gis, cx)?;
    let key = |v: &str, tr: &str| {
      if tr.is_empty() {
        v.to_string()
      } else {
        interp_key(v, tr)
      }
    };
    let (v1, v2) = (
      gis.var(&key(&n.name_test1, &n.interp_test1)),
      gis.var(&key(&n.name_test2, &n.interp_test2)),
    );
    // terms with internal tests: the blocks of the condensation
    let (q1, q2) = ((cx.internal)(&n.name_test1), (cx.internal)(&n.name_test2));
    if q1 || q2 {
      if !(n.interp_test1.is_empty() && n.interp_test2.is_empty()) {
        return None;
      }
      blocks.add(&mut gis, (v1, q1), (v2, q2), a);
      continue;
    }
    if per_point(&n.interp_test1)
      || per_point(&n.interp_test2)
      || is_im(cx, &n.name_test1)
      || is_im(cx, &n.name_test2)
    {
      gis
        .instructions
        .push(Instruction::MatAssemblyPt { t: a, v1, v2 });
      continue;
    }
    let e = gis.nb_elems as u32;
    gis.nb_elems += 1;
    let sp = gis.sp(a);
    gis
      .instructions
      .push(if sp.0 == 10 && (sp.1 == 2 || sp.1 == 3) {
        Instruction::MatAssemblyOpt10 { t: a, v1, v2, e }
      } else {
        Instruction::MatAssembly { t: a, v1, v2, e }
      });
  }
  blocks.condense(&mut gis);
  gis.fuse();
  Some(gis)
}

/// The blocks K_QQ, K_QJ, K_IQ of the condensation (GetFEM ga_compile with
/// condensation): per pair of variables (row, internal; column, internal),
/// the tensor accumulating the unweighted point tensors
#[derive(Default)]
struct CondensationBlocks(std::collections::BTreeMap<(VarRole, VarRole), u32>);

/// a variable of the instruction set and whether it is internal
type VarRole = (u32, bool);

impl CondensationBlocks {
  fn add(&mut self, gis: &mut InstructionSet, r: (u32, bool), c: (u32, bool), a: u32) {
    match self.0.get(&(r, c)) {
      Some(&b) => gis.instructions.push(Instruction::AddTo { t: b, a }),
      None => {
        let b = gis.shaped(gis.shapes[a as usize]);
        gis.instructions.push(Instruction::Copy { t: b, a });
        self.0.insert((r, c), b);
      }
    }
  }

  /// clusters of directly coupled internal variables (merged when they
  /// share one), then per cluster the condensation and the corrections
  /// of the rows coupled to it
  fn condense(&self, gis: &mut InstructionSet) {
    if self.0.is_empty() {
      return;
    }
    // union-find over the internal variables
    let mut parent: std::collections::BTreeMap<u32, u32> = Default::default();
    fn find(pa: &mut std::collections::BTreeMap<u32, u32>, x: u32) -> u32 {
      let y = *pa.entry(x).or_insert(x);
      if y == x {
        return x;
      }
      let r = find(pa, y);
      pa.insert(x, r);
      r
    }
    for &((r, qr), (c, qc)) in self.0.keys() {
      match (qr, qc) {
        (true, true) => {
          let (a, b) = (find(&mut parent, r), find(&mut parent, c));
          parent.insert(a, b);
        }
        (true, false) => {
          find(&mut parent, r);
        }
        _ => {
          find(&mut parent, c);
        }
      }
    }
    let qs: Vec<u32> = parent.keys().copied().collect();
    let mut roots: Vec<u32> = qs.iter().map(|&q| find(&mut parent, q)).collect();
    roots.sort();
    roots.dedup();
    for root in roots {
      let q: Vec<u32> = qs
        .iter()
        .copied()
        .filter(|&x| find(&mut parent, x) == root)
        .collect();
      let nq = q.len();
      let j: Vec<u32> = {
        let mut j: Vec<u32> = self
          .0
          .keys()
          .filter(|((r, qr), (_, qc))| *qr && !*qc && q.contains(r))
          .map(|(_, (c, _))| *c)
          .collect();
        j.sort();
        j.dedup();
        j
      };
      let get = |r: (u32, bool), c: (u32, bool)| self.0.get(&(r, c)).copied().unwrap_or(NONE);
      let mut cd = Cond {
        q: q.clone(),
        j: j.clone(),
        ..Default::default()
      };
      for b in 0..nq {
        for a in 0..nq {
          cd.kqq.push(get((q[a], true), (q[b], true)));
        }
      }
      for &jb in &j {
        for &qa in &q {
          let k = get((qa, true), (jb, false));
          cd.kqj.push(k);
          cd.kqjp.push(gis.shaped(Shape::Test {
            proper: 1,
            t1: qa,
            t2: jb,
          }));
        }
      }
      for &qa in &q {
        cd.rqp.push(gis.shaped(Shape::Test {
          proper: 1,
          t1: qa,
          t2: NONE,
        }));
      }
      let c = gis.conds.len() as u32;
      gis.conds.push(cd);
      gis.instructions.push(Instruction::Condense { c });
      // the rows i coupled to the cluster through K_IQ
      let mut rows: Vec<u32> = self
        .0
        .keys()
        .filter(|((_, qr), (cc, qc))| !*qr && *qc && q.contains(cc))
        .map(|((r, _), _)| *r)
        .collect();
      rows.sort();
      rows.dedup();
      for i in rows {
        let kiq: Vec<u32> = q.iter().map(|&qa| get((i, false), (qa, true))).collect();
        for (b, &jb) in j.iter().enumerate() {
          let k = gis.cond_k.len() as u32;
          gis.cond_k.push(CondK {
            c,
            jb: b as u32,
            kiq: kiq.clone(),
          });
          let t = gis.shaped(Shape::Test {
            proper: 1,
            t1: i,
            t2: jb,
          });
          gis.instructions.push(Instruction::CondenseK { t, k });
          let e = gis.nb_elems as u32;
          gis.nb_elems += 1;
          gis.instructions.push(Instruction::MatAssembly {
            t,
            v1: i,
            v2: jb,
            e,
          });
        }
        let k = gis.cond_k.len() as u32;
        gis.cond_k.push(CondK { c, jb: NONE, kiq });
        let t = gis.shaped(Shape::Test {
          proper: 1,
          t1: i,
          t2: NONE,
        });
        gis.instructions.push(Instruction::CondenseR { t, k });
        gis
          .instructions
          .push(Instruction::CondVecAssemblyPt { t, v: i });
      }
    }
  }
}

/// Tests through transformation tr assembled at each point (their target
/// element changes with the point); on the neighbor element per element
/// (GetFEM: interpolate false for neighbor_element)
/// size of the gradient index of a test function of sizes [2, q, N]: 1 if
/// absent (1D, where GetFEM does not add it)
fn n3(sizes: &[usize]) -> usize {
  sizes.get(2).copied().unwrap_or(1)
}

fn per_point(tr: &str) -> bool {
  !(tr.is_empty() || tr == crate::defs::NEIGHBOR)
}

/// Tests of an im variable: assembled at each point, at the point's index
/// (GetFEM ga_instruction_*_assembly_imd)
fn is_im(cx: &CompileCtx, name: &str) -> bool {
  cx.defs
    .var(name)
    .is_some_and(|v| v.support == Support::ImData)
}

/// A vector assembly (order 1), if all is ported.
pub(crate) fn compile_vector(trees: &[Tree], cx: &CompileCtx) -> Option<InstructionSet> {
  let rw = rewrite::all(trees);
  let trees = rw.as_deref().unwrap_or(trees);
  let mut gis = InstructionSet::default();
  gis.transformations(trees, cx)?;
  for t in trees {
    let r = t.root?;
    let n = &t.n[r];
    if n.test != 1 || n.tensor_proper_size() != 1 {
      return None;
    }
    let a = node(t, r, &mut gis, cx)?;
    if per_point(&n.interp_test1) {
      let v = gis.var(&interp_key(&n.name_test1, &n.interp_test1));
      gis
        .instructions
        .push(Instruction::VecAssemblyPt { t: a, v });
      continue;
    }
    if is_im(cx, &n.name_test1) {
      let v = gis.var(&n.name_test1);
      gis
        .instructions
        .push(Instruction::VecAssemblyPt { t: a, v });
      continue;
    }
    let v = gis.var(&if n.interp_test1.is_empty() {
      n.name_test1.clone()
    } else {
      interp_key(&n.name_test1, &n.interp_test1)
    });
    let e = gis.nb_elems as u32;
    gis.nb_elems += 1;
    gis
      .instructions
      .push(Instruction::VecAssembly { t: a, v, e });
  }
  gis.fuse();
  Some(gis)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn instructions_are_small() {
    assert!(
      std::mem::size_of::<Instruction>() <= 20,
      "{}",
      std::mem::size_of::<Instruction>()
    );
  }
}
