"""Progress bar wrapper."""
from collections.abc import Generator
from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def progress_bar(
    iterable: object,
    total: int | None = None,
    desc: str = "",
) -> Generator[tqdm, None, None]:
    with tqdm(iterable, total=total, desc=desc, unit="batch", dynamic_ncols=True) as pbar:
        yield pbar
