"""Base classes for all HistoCoreML pipelines.

Provides unified interfaces for inference and training workflows.
"""

from __future__ import annotations

import abc
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, TypeVar

from histocoreml.config import (
    C,
    InferenceResult,
    R,
    TrainingResult,
)
from histocoreml.utils import setup_logging

# Inference and training pipelines need results richer than PipelineResult:
# _log_summary reads patch_count, which only InferenceResult defines. Binding
# these keeps that a compile-time guarantee instead of an AttributeError.
IR = TypeVar("IR", bound=InferenceResult)
TR = TypeVar("TR", bound=TrainingResult)

logger = logging.getLogger(__name__)


class BasePipeline(abc.ABC, Generic[C, R]):
    """Abstract base class for all HistoCoreML pipelines.

    Generic over configuration type (C) and result type (R).

    Usage::

        with MyPipeline(cfg) as pipeline:
            results = pipeline.run(inputs)
    """

    def __init__(self, cfg: C) -> None:
        self.config = cfg
        self._timing_stats: dict[str, list[float]] = {}
        setup_logging(cfg.log_level)
        logger.info(f"Initialized {self.__class__.__name__}")

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> R | list[R]:
        """Run the pipeline. Must be implemented by subclasses."""

    def _track_time(self, operation: str, elapsed: float) -> None:
        """Track timing for an operation."""
        if operation not in self._timing_stats:
            self._timing_stats[operation] = []
        self._timing_stats[operation].append(elapsed)

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for op, times in self._timing_stats.items():
            if times:
                stats[op] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return stats

    def __enter__(self) -> BasePipeline[C, R]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


class BaseInferencePipeline(BasePipeline[C, IR], abc.ABC, Generic[C, IR]):
    """Base class for inference pipelines.

    Inference pipelines process WSI files and produce predictions.
    """

    @abc.abstractmethod
    def process_slide(self, path: Path) -> IR:
        """Process a single WSI file."""

    def run(self, wsi_paths: list[Path], *args: Any, **kwargs: Any) -> list[IR]:
        """Process a list of WSI files.

        A slide that raises is recorded as a failed result rather than aborting
        the batch, so one unreadable slide cannot lose the whole run.

        Args:
            wsi_paths: List of paths to WSI files

        Returns:
            List of result objects, one per input path and in the same order.
        """
        results: list[IR] = []

        for path in wsi_paths:
            # Started outside the try so a slide that fails still reports how
            # long it ran; otherwise every failure is timed as 0.0s and the
            # summary's total silently undercounts.
            t0 = time.perf_counter()
            try:
                result = self.process_slide(Path(path))
            except Exception as exc:
                logger.error("Error processing %s: %s", path, exc, exc_info=True)
                result = self._create_error_result(Path(path), exc)

            result.elapsed_seconds = time.perf_counter() - t0
            results.append(result)

        self._log_summary(results)
        return results

    def _create_error_result(self, path: Path, error: Exception) -> IR:
        """Build the result recorded when :meth:`process_slide` raises.

        Subclasses should override this to return their own result type; the
        fallback below carries the error but none of the subclass-specific
        fields, so callers reading e.g. ``write_result`` would not find it.
        """
        return InferenceResult(  # type: ignore[return-value]
            wsi_path=Path(path),
            elapsed_seconds=0.0,
            errors=[str(error)],
        )

    def _log_summary(self, results: list[IR]) -> None:
        """Log summary statistics."""
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        total_time = sum(r.elapsed_seconds for r in results)
        total_patches = sum(r.patch_count for r in results)

        logger.info("=" * 60)
        logger.info("Inference Summary")
        logger.info("=" * 60)
        logger.info(f"Total slides: {total}")
        logger.info(f"Successful: {success}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Total patches: {total_patches}")
        if total > 0:
            logger.info(f"Avg time per slide: {total_time/total:.1f}s")
        logger.info("=" * 60)


class BaseTrainingPipeline(BasePipeline[C, TR], abc.ABC, Generic[C, TR]):
    """Base class for training pipelines.

    Training pipelines handle the complete training workflow.
    """

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> TR:
        """Run the training pipeline."""

    def _create_error_result(self, error: Exception) -> TR:
        """Build the result recorded when training raises."""
        return TrainingResult(  # type: ignore[return-value]
            elapsed_seconds=0.0,
            errors=[str(error)],
        )


__all__ = [
    "BasePipeline",
    "BaseInferencePipeline",
    "BaseTrainingPipeline",
]
