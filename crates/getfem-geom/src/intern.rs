//! Interning of immutable catalogue objects (structures, reference convexes,
//! geometric transformations): same key, same Arc.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

pub struct Interner<K, V>(Mutex<Option<HashMap<K, Arc<V>>>>);

impl<K: Eq + Hash + Clone, V> Interner<K, V> {
  pub const fn new() -> Self {
    Interner(Mutex::new(None))
  }

  fn get(&self, k: &K) -> Option<Arc<V>> {
    self
      .0
      .lock()
      .unwrap()
      .as_ref()
      .and_then(|m| m.get(k).cloned())
  }

  /// Lock is not held while building, so builders may recurse.
  pub fn get_or_try<E>(&self, k: &K, build: impl FnOnce() -> Result<V, E>) -> Result<Arc<V>, E> {
    if let Some(v) = self.get(k) {
      return Ok(v);
    }
    let v = Arc::new(build()?);
    let mut g = self.0.lock().unwrap();
    Ok(
      g.get_or_insert_with(HashMap::new)
        .entry(k.clone())
        .or_insert(v)
        .clone(),
    )
  }

  pub fn get_or(&self, k: &K, build: impl FnOnce() -> V) -> Arc<V> {
    self
      .get_or_try::<std::convert::Infallible>(k, || Ok(build()))
      .unwrap_or_else(|e| match e {})
  }
}

impl<K: Eq + Hash + Clone, V> Default for Interner<K, V> {
  fn default() -> Self {
    Self::new()
  }
}
