"""Progress bar utilities for training and inference.

Provides tqdm-based progress bars with consistent formatting
for use across the HistoCoreML codebase.
"""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm


@contextmanager
def progress_bar(
    iterable: Iterable[object],
    total: int | None = None,
    desc: str = "",
    unit: str = "batch",
    **kwargs: Any,
) -> Generator[tqdm, None, None]:
    """Context manager for tqdm progress bar.

    Args:
        iterable: Iterable to wrap with progress bar
        total: Total number of items (inferred from iterable if None)
        desc: Description shown on progress bar
        unit: Unit label (e.g., "batch", "patch", "slide")
        **kwargs: Additional arguments passed to tqdm

    Usage:
        with progress_bar(dataloader, desc="Training") as pbar:
            for batch in pbar:
                # process batch
                pbar.set_postfix({"loss": loss.item()})
    """
    with tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        **kwargs,
    ) as pbar:
        yield pbar


def create_progress_bar(
    total: int,
    desc: str = "",
    unit: str = "batch",
    **kwargs: Any,
) -> tqdm:
    """Create a tqdm progress bar for manual iteration.

    Args:
        total: Total number of items
        desc: Description shown on progress bar
        unit: Unit label
        **kwargs: Additional arguments passed to tqdm

    Usage:
        pbar = create_progress_bar(len(dataset), desc="Processing")
        for item in dataset:
            # process item
            pbar.update(1)
        pbar.close()
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        **kwargs,
    )
