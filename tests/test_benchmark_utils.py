"""Tests for benchmarks.utils (TimingTree and metadata)."""

import json
import time

from benchmarks.utils import TimingTree, collect_metadata


class TestTimingTree:
    def test_single_block(self):
        tt = TimingTree()
        with tt.time("total"):
            time.sleep(0.01)
        assert tt.root is not None
        assert tt.root.name == "total"
        assert tt.root.elapsed >= 0.01

    def test_nested_blocks(self):
        tt = TimingTree()
        with tt.time("total"):
            with tt.time("a"):
                time.sleep(0.01)
            with tt.time("b"):
                time.sleep(0.01)
        assert len(tt.root.children) == 2
        assert tt.root.children[0].name == "a"
        assert tt.root.children[1].name == "b"

    def test_deep_nesting(self):
        tt = TimingTree()
        with tt.time("l0"):
            with tt.time("l1"):
                with tt.time("l2"):
                    time.sleep(0.01)
        assert tt.root.children[0].children[0].name == "l2"

    def test_to_dict(self):
        tt = TimingTree()
        with tt.time("total"):
            with tt.time("step"):
                pass
        d = tt.to_dict()
        assert d["name"] == "total"
        assert "elapsed_s" in d
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "step"

    def test_save(self, tmp_path):
        tt = TimingTree()
        with tt.time("total"):
            pass
        out = tmp_path / "timing.json"
        tt.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["name"] == "total"

    def test_empty_tree(self):
        tt = TimingTree()
        assert tt.root is None
        assert tt.to_dict() == {}


class TestCollectMetadata:
    def test_returns_dict(self):
        meta = collect_metadata()
        assert isinstance(meta, dict)

    def test_required_keys(self):
        meta = collect_metadata()
        expected_keys = {
            "git_hash",
            "git_branch",
            "git_dirty",
            "hostname",
            "python_version",
            "torch_version",
            "gpu_name",
            "gpu_count",
            "platform",
        }
        assert expected_keys <= set(meta.keys())

    def test_git_hash_nonempty(self):
        meta = collect_metadata()
        # We're in a git repo, so this should be populated
        assert len(meta["git_hash"]) > 0

    def test_python_version_format(self):
        meta = collect_metadata()
        parts = meta["python_version"].split(".")
        assert len(parts) == 3
