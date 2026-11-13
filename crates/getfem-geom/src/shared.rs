//! Shared handle to one object (SPEC 3, ownership): every clone of the
//! handle sees the changes made through any other. Readers take a snapshot
//! that stays consistent while they use it; a change replaces the current
//! version (copy-on-write if a snapshot is still in use).

use std::sync::{Arc, RwLock};

pub struct Shared<T>(Arc<RwLock<Arc<T>>>);

impl<T> Clone for Shared<T> {
  fn clone(&self) -> Self {
    Shared(self.0.clone())
  }
}

impl<T: Clone> Shared<T> {
  pub fn new(value: T) -> Self {
    Shared(Arc::new(RwLock::new(Arc::new(value))))
  }

  /// The current version, unaffected by later changes.
  pub fn snapshot(&self) -> Arc<T> {
    self.0.read().unwrap_or_else(|e| e.into_inner()).clone()
  }

  /// Change the object; the result of f is returned.
  pub fn modify<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
    let mut g = self.0.write().unwrap_or_else(|e| e.into_inner());
    f(Arc::make_mut(&mut g))
  }

  /// Same object (not only equal)?
  pub fn same(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.0, &other.0)
  }

  /// The object's identity (same for its clones), never 0
  pub fn id(&self) -> usize {
    Arc::as_ptr(&self.0) as *const () as usize
  }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let g = self.0.read().unwrap_or_else(|e| e.into_inner());
    f.debug_tuple("Shared").field(&**g).finish()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn changes_are_shared_snapshots_are_stable() {
    let a = Shared::new(vec![1, 2]);
    let b = a.clone();
    let before = a.snapshot();
    b.modify(|v| v.push(3));
    assert_eq!(*a.snapshot(), vec![1, 2, 3]);
    assert_eq!(*before, vec![1, 2]);
    assert!(a.same(&b) && !a.same(&Shared::new(vec![1, 2, 3])));
  }
}
