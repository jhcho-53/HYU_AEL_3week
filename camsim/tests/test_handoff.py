import json

import pytest

from camsim import config, handoff


def test_export_checkpoint_writes_verified_copy_and_manifest(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"example checkpoint bytes")
    destination = tmp_path / "drive"

    manifest = handoff.export_checkpoint(source, destination, config.load(), "abc123")

    assert (destination / "model.pt").read_bytes() == source.read_bytes()
    assert manifest["sha256"] == handoff.sha256_file(destination / "model.pt")
    assert manifest["bytes"] == source.stat().st_size
    assert manifest["git_commit"] == "abc123"
    assert manifest["config"]["bev"]["resolution_m"] == config.load().bev.resolution_m
    assert json.loads((destination / "checkpoint.json").read_text()) == manifest


def test_export_checkpoint_requires_nonempty_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        handoff.export_checkpoint(tmp_path / "missing.pt", tmp_path / "drive", config.load(), "abc123")
    empty = tmp_path / "empty.pt"
    empty.touch()
    with pytest.raises(ValueError, match="empty checkpoint"):
        handoff.export_checkpoint(empty, tmp_path / "drive", config.load(), "abc123")
