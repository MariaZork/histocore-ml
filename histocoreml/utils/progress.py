"""Progress bar wrapper."""
from contextlib import contextmanager
from tqdm import tqdm


@contextmanager
def progress_bar(iterable, total=None, desc=""):
    with tqdm(iterable, total=total, desc=desc, unit="batch", dynamic_ncols=True) as pbar:
        yield pbar
