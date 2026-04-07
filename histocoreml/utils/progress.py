"""Progress bar wrapper."""
from collections.abc import Generator, Iterable
from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def progress_bar(
    iterable: Iterable[object],
    total: int | None = None,
    desc: str = "",
) -> Generator[tqdm, None, None]:
    with tqdm(iterable, total=total, desc=desc, unit="batch", dynamic_ncols=True) as pbar:
        yield pbar
