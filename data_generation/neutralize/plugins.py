"""Load a user-supplied function by `module:function` or `path/to/file.py:function`.

Both the eval metric and the similarity score are things you will want to swap
without touching this package. Neither is hard-wired: the driver takes a spec
string, this resolves it, and the result is called as a plain function.
"""

import importlib
import importlib.util
import sys
from pathlib import Path


def load_callable(spec):
    """`pkg.mod:func` or `scripts/my_metric.py:func` -> the function."""
    if ":" not in spec:
        raise ValueError(
            f"expected 'module:function' or 'file.py:function', got {spec!r}")
    target, name = spec.rsplit(":", 1)

    if target.endswith(".py") or "/" in target:
        path = Path(target).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        mod_name = f"_plugin_{path.stem}"
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = mod
        spec_obj.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)

    fn = getattr(mod, name, None)
    if fn is None:
        raise AttributeError(f"{target} has no attribute {name!r}")
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn
