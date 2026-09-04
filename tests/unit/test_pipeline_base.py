"""Tests for the shared pipeline base classes."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from histocoreml.config import PipelineConfig, SegmentationInferenceResult
from histocoreml.pipelines.base import BaseInferencePipeline


class _StubPipeline(BaseInferencePipeline):
    """Sleeps for a fixed time, failing on any slide named ``bad*``."""

    DELAY = 0.05

    def process_slide(self, path: Path) -> SegmentationInferenceResult:
        time.sleep(self.DELAY)
        if path.name.startswith("bad"):
            raise RuntimeError("synthetic failure")
        return SegmentationInferenceResult(
            wsi_path=path, patch_count=3, write_result=None, mask_path=None
        )

    def _create_error_result(self, path: Path, error: Exception) -> SegmentationInferenceResult:
        return SegmentationInferenceResult(
            wsi_path=path,
            elapsed_seconds=0.0,  # deliberately wrong; run() must override it
            errors=[str(error)],
            patch_count=0,
            write_result=None,
            mask_path=None,
        )


@pytest.fixture
def pipeline() -> _StubPipeline:
    return _StubPipeline(PipelineConfig())


class TestRunTiming:
    def test_successful_slide_is_timed(self, pipeline: _StubPipeline):
        (result,) = pipeline.run([Path("good.svs")])

        assert result.success
        assert result.elapsed_seconds >= _StubPipeline.DELAY

    def test_failed_slide_is_also_timed(self, pipeline: _StubPipeline):
        """A failure used to report 0.0s, so summary totals undercounted."""
        (result,) = pipeline.run([Path("bad.svs")])

        assert not result.success
        assert result.elapsed_seconds >= _StubPipeline.DELAY

    def test_summary_total_includes_failures(self, pipeline: _StubPipeline, caplog):
        with caplog.at_level(logging.INFO, logger="histocoreml.pipelines.base"):
            pipeline.run([Path("bad1.svs"), Path("bad2.svs")])

        total_line = next(m for m in caplog.messages if m.startswith("Total time"))
        assert total_line != "Total time: 0.0s"


class TestRunResilience:
    def test_one_failure_does_not_abort_the_batch(self, pipeline: _StubPipeline):
        results = pipeline.run([Path("good1.svs"), Path("bad.svs"), Path("good2.svs")])

        assert len(results) == 3
        assert [r.success for r in results] == [True, False, True]

    def test_results_preserve_input_order(self, pipeline: _StubPipeline):
        paths = [Path("bad.svs"), Path("good.svs")]

        results = pipeline.run(paths)

        assert [r.wsi_path for r in results] == paths

    def test_error_message_is_recorded(self, pipeline: _StubPipeline):
        (result,) = pipeline.run([Path("bad.svs")])

        assert result.errors == ["synthetic failure"]

    def test_empty_input_returns_empty(self, pipeline: _StubPipeline):
        assert pipeline.run([]) == []


class TestSummaryStatistics:
    def test_patch_counts_are_summed(self, pipeline: _StubPipeline, caplog):
        with caplog.at_level(logging.INFO, logger="histocoreml.pipelines.base"):
            pipeline.run([Path("good1.svs"), Path("good2.svs")])

        assert "Total patches: 6" in caplog.messages
        assert "Successful: 2" in caplog.messages
        assert "Failed: 0" in caplog.messages

    def test_context_manager_protocol(self, pipeline: _StubPipeline):
        with pipeline as entered:
            assert entered is pipeline
