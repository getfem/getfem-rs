//! Columns spanning the range of a sparse matrix:
//! column-pivoted modified Gram-Schmidt on the normalized columns
//! themselves (not on their Gram matrix, which squares the conditioning),
//! per group of columns connected through shared rows.

use std::collections::BTreeMap;

use crate::Csr;

/// Indices (increasing) of linearly independent columns of b spanning its
/// range: a column is kept while its part orthogonal to the kept ones has a
/// norm above eps (columns normalized first; columns below eps times the
/// largest are zero).
pub fn range_basis(b: &Csr<f64>, eps: f64) -> Vec<usize> {
  range_basis_after(b, 0, eps)
}

/// As range_basis, with the first `fixed` columns (independent) kept first:
/// the other columns kept are independent of them too.
pub fn range_basis_after(b: &Csr<f64>, fixed: usize, eps: f64) -> Vec<usize> {
  let bt = b.transpose();
  let inf = |j: usize| bt.row(j).fold(0.0f64, |m, (_, v)| m.max(v.abs()));
  let norm_max = (0..b.ncols).map(inf).fold(0.0f64, f64::max);
  let cols: Vec<Vec<(usize, f64)>> = (0..b.ncols)
    .map(|j| {
      if inf(j) <= eps * norm_max {
        return vec![];
      }
      let c: Vec<(usize, f64)> = bt.row(j).filter(|(_, v)| *v != 0.0).collect();
      let n = c.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
      c.into_iter().map(|(i, v)| (i, v / n)).collect()
    })
    .collect();
  // groups of columns sharing rows (union-find)
  let mut parent: Vec<usize> = (0..b.ncols).collect();
  fn find(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
      p[i] = p[p[i]];
      i = p[i];
    }
    i
  }
  let mut owner = vec![usize::MAX; b.nrows];
  for (j, c) in cols.iter().enumerate() {
    for &(i, _) in c {
      if owner[i] == usize::MAX {
        owner[i] = j;
      } else {
        let (a, r) = (find(&mut parent, owner[i]), find(&mut parent, j));
        parent[a.max(r)] = a.min(r);
      }
    }
  }
  let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
  for (j, c) in cols.iter().enumerate() {
    if !c.is_empty() {
      let r = find(&mut parent, j);
      groups.entry(r).or_default().push(j);
    }
  }
  let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
  let mut kept = vec![];
  for g in groups.values() {
    // dense columns on the rows of the group
    let mut rows: Vec<usize> = g
      .iter()
      .flat_map(|&j| cols[j].iter().map(|e| e.0))
      .collect();
    rows.sort_unstable();
    rows.dedup();
    let pos: BTreeMap<usize, usize> = rows.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let mut w: Vec<Vec<f64>> = g
      .iter()
      .map(|&j| {
        let mut d = vec![0.0; rows.len()];
        for &(i, v) in &cols[j] {
          d[pos[&i]] = v;
        }
        d
      })
      .collect();
    let project = |w: &mut Vec<Vec<f64>>, c: usize, n: f64, others: &[usize]| {
      let q: Vec<f64> = w[c].iter().map(|x| x / n).collect();
      for &o in others {
        let r: f64 = q.iter().zip(&w[o]).map(|(a, b)| a * b).sum();
        w[o].iter_mut().zip(&q).for_each(|(x, qi)| *x -= r * qi);
      }
    };
    // the fixed columns first, in order
    let mut left: Vec<usize> = (0..g.len()).filter(|&c| g[c] >= fixed).collect();
    for c in 0..g.len() {
      if g[c] < fixed {
        kept.push(g[c]);
        let n = norm(&w[c]);
        let others: Vec<usize> = (c + 1..g.len()).collect();
        project(&mut w, c, n, &others);
      }
    }
    while !left.is_empty() {
      // the largest remaining part (lowest index on ties)
      let (k, nmax) = left
        .iter()
        .enumerate()
        .map(|(k, &c)| (k, norm(&w[c])))
        .fold((0, -1.0), |a, b| if b.1 > a.1 { b } else { a });
      if nmax <= eps {
        break;
      }
      let c = left.remove(k);
      kept.push(g[c]);
      project(&mut w, c, nmax, &left);
    }
  }
  kept.sort_unstable();
  kept
}
