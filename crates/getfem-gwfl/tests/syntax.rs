//! GWFL syntax: a lone equal sign is invalid.

use getfem_gwfl::parser::{MacroDict, parse};

/// a lone '=' is not ':=' (GetFEM ga_get_token: GA_INVALID)
#[test]
fn lone_equal_is_invalid() {
  let dict = MacroDict::default();
  assert!(parse("Def f = 2*x", &dict).is_err());
  assert!(parse("a = b", &dict).is_err());
  assert!(parse("Def f := 2*x; f", &dict).is_ok());
}
