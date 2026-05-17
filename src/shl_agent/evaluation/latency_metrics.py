import time
from typing import Callable, Any, Tuple


def timeit(fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
