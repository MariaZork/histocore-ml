"""Shared pytest fixtures for HistoCoreML test suite."""

from __future__ import annotations

import numpy as np
import pytest

from histocoreml.preprocessing.patch_coord import PatchCoord


def make_coord(row: int = 0, col: int = 0, patch_size: int = 64) -> PatchCoord:
    return PatchCoord(
        x=col * patch_size,
        y=row * patch_size,
        level=0,
        patch_size=patch_size,
        col_idx=col,
        row_idx=row,
    )


def make_rgb_patch(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 200, size=(h, w, 3), dtype=np.uint8)  # tissue-like (< 230)


def make_binary_mask(h: int = 64, w: int = 64, ratio: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((h, w)) < ratio).astype(np.uint8)


@pytest.fixture
def coord() -> PatchCoord:
    return make_coord()


@pytest.fixture
def rgb_patch() -> np.ndarray:
    return make_rgb_patch()


@pytest.fixture
def binary_mask() -> np.ndarray:
    return make_binary_mask()


@pytest.fixture
def all_zero_mask() -> np.ndarray:
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def all_ones_mask() -> np.ndarray:
    return np.ones((64, 64), dtype=np.uint8)
