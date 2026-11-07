"""Focused regressions for the v0.36 plotting and Trainer review lane."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.plotting.animation_quality import verify_animation_encoding
from sim.review.animation_planning import ReviewAnimationSpec
from sim.review.animation_rendering import ReviewAnimationArtifact
from sim.review.plotting import ReviewPlotArtifact, ReviewPlotSpec


class _FakeWriter:
    def __init__(self, root: Path, logical_root: Path) -> None:
        self.path = root
        self.logical_root = logical_root
        self.calls: list[tuple[str, Path]] = []
        self.files: dict[Path, bytes] = {}

    def file_record(self, relative: str | Path) -> dict[str, object]:
        relative = Path(relative)
        self.calls.append(("file_record", relative))
        if relative not in self.files:
            raise FileNotFoundError(relative)
        payload = self.files[relative]
        return {"bytes": len(payload), "sha256": "fixture"}

    def read_bytes(self, relative: str | Path, *, max_bytes: int | None = None) -> bytes:
        relative = Path(relative)
        self.calls.append(("read_bytes", relative))
        payload = self.files[relative]
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError("fixture exceeds read limit")
        return payload

    def write_bytes(self, relative: str | Path, payload: bytes, **_kwargs: object) -> Path:
        relative = Path(relative)
        self.calls.append(("write_bytes", relative))
        if relative in self.files:
            raise FileExistsError(relative)
        self.files[relative] = bytes(payload)
        return self.path / relative

    def update_existing_bytes(self, relative: str | Path, payload: bytes) -> Path:
        relative = Path(relative)
        self.calls.append(("update_existing_bytes", relative))
        if relative not in self.files:
            raise FileNotFoundError(relative)
        self.files[relative] = bytes(payload)
        return self.path / relative


def test_animation_receipt_decodes_every_gif_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image

    target = tmp_path / "movie.gif"
    target.write_bytes(b"fixture")

    class FakeDecoded:
        size = (20, 12)

    class FakeGif:
        size = (20, 12)
        n_frames = 4
        info = {"duration": 100.0}

        def __init__(self) -> None:
            self.current = -1
            self.converted: list[int] = []

        def __enter__(self) -> FakeGif:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def seek(self, index: int) -> None:
            self.current = int(index)

        def convert(self, _mode: str) -> FakeDecoded:
            self.converted.append(self.current)
            return FakeDecoded()

    fake = FakeGif()
    monkeypatch.setattr(Image, "open", lambda _path: fake)

    report = verify_animation_encoding(target, expected_frames=4, expected_fps=10.0)

    assert report.decode_ok is True
    assert fake.converted == [0, 1, 2, 3]


def test_animation_receipt_rejects_truncated_gif(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image

    target = tmp_path / "truncated.gif"
    target.write_bytes(b"fixture")

    class FakeDecoded:
        size = (20, 12)

    class FakeGif:
        size = (20, 12)
        n_frames = 1
        info = {"duration": 100.0}

        def __enter__(self) -> FakeGif:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def seek(self, _index: int) -> None:
            return None

        def convert(self, _mode: str) -> FakeDecoded:
            return FakeDecoded()

    monkeypatch.setattr(Image, "open", lambda _path: FakeGif())

    report = verify_animation_encoding(target, expected_frames=4, expected_fps=10.0)

    assert report.frame_count == 1
    assert report.decode_ok is False


def test_stale_protected_plot_does_not_unlink_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sim.review.plot_planning as planning

    workspace = SimpleNamespace(
        output_dir=tmp_path,
        evidence_identity=lambda: {"review": "same"},
        query=lambda *_args, **_kwargs: object(),
    )
    path = tmp_path / "replacement.png"
    path.write_bytes(b"replacement")
    spec = ReviewPlotSpec(sql="SELECT 1", x_column="x", y_columns=["y"])
    artifact = ReviewPlotArtifact(
        artifact_id="replacement",
        path=path,
        relative_path="replacement.png",
        row_count=1,
        truncated=False,
        spec=spec,
    )
    ids = iter(("plan", "stale"))
    monkeypatch.setattr(planning.ReviewWorkspace, "open", lambda _path: workspace)
    monkeypatch.setattr(planning, "review_plot_plan_id", lambda *_args, **_kwargs: next(ids))
    monkeypatch.setattr(planning, "_dry_run_from_result", lambda *_args, **_kwargs: {"truncated": False})
    monkeypatch.setattr(planning, "_save_review_plot_from_result", lambda *_args, **_kwargs: artifact)

    with pytest.raises(ValueError, match="review store changed"):
        planning.render_review_plot(
            tmp_path,
            spec,
            plot_plan_id="plan",
            path=path,
            writer=object(),
        )

    assert path.read_bytes() == b"replacement"


def test_stale_protected_animation_does_not_unlink_companions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sim.review.animation_planning as planning
    import sim.review.animation_rendering as rendering

    workspace = SimpleNamespace(output_dir=tmp_path, evidence_identity=lambda: {"review": "same"})
    recipe = object()
    result = SimpleNamespace(truncated=False)
    movie = tmp_path / "replacement.gif"
    contact = tmp_path / "replacement.contact-sheet.png"
    receipt = tmp_path / "replacement.quality.json"
    for path in (movie, contact, receipt):
        path.write_bytes(b"replacement")
    spec = ReviewAnimationSpec(recipe_id="relative_position_ric_2d", artifact_id="replacement", file_format="gif")
    artifact = ReviewAnimationArtifact(
        artifact_id="replacement",
        path=movie,
        relative_path=movie.name,
        contact_sheet_path=contact,
        quality_receipt_path=receipt,
        row_count=1,
        truncated=False,
        spec=spec,
        qa={},
    )
    ids = iter(("plan", "stale"))
    monkeypatch.setattr(planning.ReviewWorkspace, "open", lambda _path: workspace)
    monkeypatch.setattr(planning, "load_review_animation_evidence", lambda *_args, **_kwargs: (recipe, result))
    monkeypatch.setattr(planning, "_review_animation_plan_id_from_evidence", lambda *_args, **_kwargs: next(ids))
    monkeypatch.setattr(rendering, "render_review_animation_artifact", lambda *_args, **_kwargs: artifact)

    with pytest.raises(ValueError, match="review store changed"):
        planning.render_review_animation(
            tmp_path,
            spec,
            animation_plan_id="plan",
            path=movie,
            writer=object(),
        )

    assert all(path.read_bytes() == b"replacement" for path in (movie, contact, receipt))


def test_protected_plot_index_uses_writer_descriptor_reads(tmp_path: Path) -> None:
    from sim.review.plotting import record_generated_artifact

    logical = tmp_path / "logical"
    physical = tmp_path / "physical"
    writer = _FakeWriter(physical, logical)
    writer.files[Path("review/generated_artifacts.json")] = b'{"artifacts": []}\n'
    workspace = SimpleNamespace(
        output_dir=logical,
        query=lambda *_args, **_kwargs: SimpleNamespace(rows=[]),
        evidence_identity=lambda: {"review": "same"},
    )
    spec = ReviewPlotSpec(sql="SELECT 1", x_column="x", y_columns=["y"])
    artifact = ReviewPlotArtifact(
        artifact_id="plot",
        path=physical / "review" / "figures" / "plot.png",
        relative_path="review/figures/plot.png",
        row_count=1,
        truncated=False,
        spec=spec,
    )

    record_generated_artifact(workspace, artifact, writer=writer, logical_output_dir=logical)

    assert ("file_record", Path("review/generated_artifacts.json")) in writer.calls
    assert ("read_bytes", Path("review/generated_artifacts.json")) in writer.calls
    assert Path("review/generated_artifacts.json") in writer.files


def test_protected_animation_index_uses_writer_descriptor_reads(tmp_path: Path) -> None:
    from sim.review.animation_rendering import record_generated_animation

    logical = tmp_path / "logical"
    physical = tmp_path / "physical"
    writer = _FakeWriter(physical, logical)
    writer.files[Path("review/generated_artifacts.json")] = b'{"artifacts": []}\n'
    workspace = SimpleNamespace(
        output_dir=logical,
        review_dir=logical / "review",
        evidence_identity=lambda: {"review": "same"},
    )
    spec = ReviewAnimationSpec(recipe_id="relative_position_ric_2d", artifact_id="movie", file_format="gif")
    artifact = ReviewAnimationArtifact(
        artifact_id="movie",
        path=physical / "review" / "animations" / "movie.gif",
        relative_path="review/animations/movie.gif",
        contact_sheet_path=physical / "review" / "animations" / "movie.contact-sheet.png",
        quality_receipt_path=physical / "review" / "animations" / "movie.quality.json",
        row_count=1,
        truncated=False,
        spec=spec,
        qa={},
        logical_contact_sheet_path=logical / "review" / "animations" / "movie.contact-sheet.png",
        logical_quality_receipt_path=logical / "review" / "animations" / "movie.quality.json",
    )

    record_generated_animation(workspace, artifact, recipe=SimpleNamespace(
        sql="SELECT 1", recipe_id="relative_position_ric_2d", recipe_version=1, renderer_id="generic"
    ), writer=writer, logical_output_dir=logical)

    assert ("file_record", Path("review/generated_artifacts.json")) in writer.calls
    assert Path("review/generated_artifacts.json") in writer.files


def test_mcp_report_writer_mapping_is_lexical(tmp_path: Path) -> None:
    from integrations.oel_mcp import reporting

    logical = tmp_path / "logical"
    physical = tmp_path / "physical"
    writer = _FakeWriter(physical, logical)

    assert reporting._logical_output_dir(
        physical,
        writer=writer,
        logical_output_dir=logical,
    ) == logical
    written = reporting._write_output_bytes(physical / "report.json", b"{}", writer=writer)

    assert written == physical / "report.json"
    assert Path("report.json") in writer.files


def test_validation_doc_uses_current_challenge_identity() -> None:
    doc = Path(__file__).parents[2] / "web" / "rpo-trainer-preview" / "docs" / "competition-validation.md"
    packet = json.loads(re.search(r"```json\n(.*?)\n```", doc.read_text(encoding="utf-8"), re.S).group(1))
    fixture = json.loads(
        (
            Path(__file__).parents[2]
            / "web"
            / "rpo-trainer-preview"
            / "fixtures"
            / "default-challenge-record.json"
        ).read_text(encoding="utf-8")
    )
    assert packet["challenge_id"] == fixture["challenge_id"]
    assert packet["config_hash"] == fixture["config_hash"]
