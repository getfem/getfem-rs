# the examples importable by the tests (as the Rust tests' #[path])
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../examples"))
