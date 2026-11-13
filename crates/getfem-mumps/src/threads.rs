//! The threads of MUMPS: those of its BLAS, and of its OpenMP if built with
//! it. Process-wide, independent of the assembly's (getfem::set_num_threads);
//! 1 by default, as GetFEM's GETFEM_FORCE_SINGLE_THREAD_BLAS. GetFEM-rs has
//! no BLAS of its own: the BLAS is MUMPS', the library of dgemm_.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

use getfem_linalg::{Error, Result};

unsafe extern "C" {
  fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
  fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
  fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
}

#[repr(C)]
struct DlInfo {
  fname: *const c_char,
  fbase: *mut c_void,
  sname: *const c_char,
  saddr: *mut c_void,
}

// glibc
const RTLD_LAZY: c_int = 1;
const RTLD_NOLOAD: c_int = 4;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);

/// The number of threads of MUMPS (its BLAS, its OpenMP), from its next
/// call on; n >= 1. With MPI on several ranks only 1, more is an error.
pub fn set_num_threads(n: usize) -> Result<()> {
  assert!(n >= 1, "MUMPS: at least one thread");
  if n > 1 && getfem_mpi::size() > 1 {
    return Err(Error::InvalidArgument(format!(
      "getfem::mumps::set_num_threads({n}): MPI on several ranks runs one \
       thread per rank"
    )));
  }
  NUM_THREADS.store(n, Ordering::Relaxed);
  Ok(())
}

pub fn num_threads() -> usize {
  NUM_THREADS.load(Ordering::Relaxed)
}

/// The BLAS of MUMPS and its thread control
#[derive(Clone, Debug, Default)]
pub struct BlasInfo {
  /// the library of dgemm_
  pub path: Option<String>,
  /// "OpenBLAS", "BLIS", "MKL"; None: no known thread control
  pub name: Option<&'static str>,
  /// OpenBLAS: "serial", "pthread", "openmp"
  pub parallel: Option<&'static str>,
  /// its threads now
  pub num_threads: Option<usize>,
  /// an OpenMP runtime loaded (MUMPS' OpenMP threads follow num_threads())
  pub openmp: bool,
}

impl fmt::Display for BlasInfo {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "BLAS {}", self.path.as_deref().unwrap_or("not found"))?;
    match self.name {
      Some(n) => write!(f, ", {n}")?,
      None => write!(f, ", no known thread control")?,
    }
    if let Some(p) = self.parallel {
      write!(f, " ({p})")?;
    }
    if let Some(t) = self.num_threads {
      write!(f, ", {t} threads")?;
    }
    if self.openmp {
      write!(f, ", OpenMP runtime loaded")?;
    }
    Ok(())
  }
}

type SetFn = extern "C" fn(c_int);
type GetFn = extern "C" fn() -> c_int;

/// The thread control of the BLAS (its set and get), and OpenMP's set
#[derive(Default)]
struct Controls {
  path: Option<String>,
  name: Option<&'static str>,
  set: Option<SetFn>,
  get: Option<GetFn>,
  parallel: Option<GetFn>,
  omp_set: Option<SetFn>,
}

/// symbol of handle (and its dependencies), or of the global scope (null)
fn sym(handle: *mut c_void, s: &CStr) -> *mut c_void {
  // SAFETY: dlsym with a handle from dlopen or null (RTLD_DEFAULT)
  unsafe { dlsym(handle, s.as_ptr()) }
}

fn controls() -> &'static Controls {
  static C: OnceLock<Controls> = OnceLock::new();
  C.get_or_init(|| {
    let mut c = Controls::default();
    let dgemm = sym(std::ptr::null_mut(), c"dgemm_");
    if dgemm.is_null() {
      return c;
    }
    let mut info = DlInfo {
      fname: std::ptr::null(),
      fbase: std::ptr::null_mut(),
      sname: std::ptr::null(),
      saddr: std::ptr::null_mut(),
    };
    // SAFETY: dladdr of a loaded symbol; fname, if set, a C string of the
    // loader; dlopen with RTLD_NOLOAD only returns the loaded library
    let lib = unsafe {
      if dladdr(dgemm, &mut info) == 0 || info.fname.is_null() {
        return c;
      }
      dlopen(info.fname, RTLD_LAZY | RTLD_NOLOAD)
    };
    // SAFETY: as above
    c.path = Some(
      unsafe { CStr::from_ptr(info.fname) }
        .to_string_lossy()
        .into_owned(),
    );
    if lib.is_null() {
      return c;
    }
    // the controls of the library of dgemm_ (or of its dependencies)
    let known: [(&str, &CStr, &CStr); 3] = [
      (
        "OpenBLAS",
        c"openblas_set_num_threads",
        c"openblas_get_num_threads",
      ),
      (
        "BLIS",
        c"bli_thread_set_num_threads",
        c"bli_thread_get_num_threads",
      ),
      ("MKL", c"MKL_Set_Num_Threads", c"MKL_Get_Max_Threads"),
    ];
    for (name, set, get) in known {
      let (s, g) = (sym(lib, set), sym(lib, get));
      if !s.is_null() && !g.is_null() {
        // SAFETY: the C signatures of these libraries: void f(int), int f()
        unsafe {
          c.set = Some(std::mem::transmute::<*mut c_void, SetFn>(s));
          c.get = Some(std::mem::transmute::<*mut c_void, GetFn>(g));
        }
        c.name = Some(name);
        break;
      }
    }
    let p = sym(lib, c"openblas_get_parallel");
    if c.name == Some("OpenBLAS") && !p.is_null() {
      // SAFETY: int openblas_get_parallel(void)
      c.parallel = Some(unsafe { std::mem::transmute::<*mut c_void, GetFn>(p) });
    }
    let o = sym(std::ptr::null_mut(), c"omp_set_num_threads");
    if !o.is_null() {
      // SAFETY: void omp_set_num_threads(int)
      c.omp_set = Some(unsafe { std::mem::transmute::<*mut c_void, SetFn>(o) });
    }
    c
  })
}

/// The BLAS of MUMPS, its thread control and threads now
pub fn blas_info() -> BlasInfo {
  let c = controls();
  BlasInfo {
    path: c.path.clone(),
    name: c.name,
    parallel: c.parallel.map(|p| match p() {
      0 => "serial",
      1 => "pthread",
      _ => "openmp",
    }),
    num_threads: c.get.map(|g| g() as usize),
    openmp: c.omp_set.is_some(),
  }
}

/// num_threads() to the BLAS and OpenMP, before each MUMPS call; with more
/// than one thread, a warning (once) if they cannot follow
pub(crate) fn apply() {
  // MPI on several ranks: single-threaded ranks
  let n = match getfem_mpi::size() {
    1 => num_threads(),
    _ => {
      if num_threads() > 1 {
        static MPI: Once = Once::new();
        MPI.call_once(|| {
          eprintln!(
            "GetFEM-rs warning: MPI on several ranks: MUMPS on one thread per rank \
             (the threads set before ignored)"
          )
        });
      }
      1
    }
  };
  let c = controls();
  let ni = c_int::try_from(n).unwrap_or(c_int::MAX);
  if let Some(set) = c.set {
    set(ni);
  }
  if let Some(set) = c.omp_set {
    set(ni);
  }
  if c.set.is_some() || c.omp_set.is_some() {
    static ENV: Once = Once::new();
    ENV.call_once(|| {
      let vars = [
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "OMP_NUM_THREADS",
      ];
      let vars = vars
        .into_iter()
        .chain(["BLIS_NUM_THREADS", "MKL_NUM_THREADS"]);
      let set: Vec<String> = vars
        .filter_map(|v| std::env::var(v).ok().map(|x| format!("{v}={x}")))
        .collect();
      if !set.is_empty() {
        eprintln!(
          "GetFEM-rs warning: {} overridden in MUMPS' calls: {n} threads \
           (getfem::mumps::set_num_threads)",
          set.join(", ")
        );
      }
    });
  }
  if n > 1 {
    static WARN: Once = Once::new();
    WARN.call_once(|| {
      let i = blas_info();
      let why = if i.name.is_none() {
        Some("not OpenBLAS, BLIS or MKL: its threads, if any, by its own variable")
      } else if i.parallel == Some("serial") {
        Some("it is serial")
      } else if i.num_threads != Some(n) {
        Some("it did not take the threads")
      } else {
        None
      };
      if let Some(why) = why {
        eprintln!("GetFEM-rs warning: MUMPS on {n} threads, but {i}: {why}");
      }
    });
  }
}
