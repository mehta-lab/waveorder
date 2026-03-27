"""Shared utilities for benchmarks: timing, metadata, and fixtures."""

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- TimingTree ---


@dataclass
class TimingNode:
    """A single timing measurement, possibly with children."""

    name: str
    elapsed: float = 0.0
    children: list["TimingNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d = {"name": self.name, "elapsed_s": round(self.elapsed, 4)}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


class TimingTree:
    """Nested timing context manager.

    Examples
    --------
    >>> tt = TimingTree()
    >>> with tt.time("total"):
    ...     with tt.time("step_a"):
    ...         pass
    ...     with tt.time("step_b"):
    ...         pass
    >>> tt.root.name
    'total'
    >>> len(tt.root.children)
    2
    """

    def __init__(self):
        self.root: TimingNode | None = None
        self._stack: list[TimingNode] = []

    @contextmanager
    def time(self, name: str):
        """Time a named block. Blocks can be nested."""
        node = TimingNode(name=name)
        if self._stack:
            self._stack[-1].children.append(node)
        else:
            self.root = node
        self._stack.append(node)
        start = time.perf_counter()
        try:
            yield node
        finally:
            node.elapsed = time.perf_counter() - start
            self._stack.pop()

    def to_dict(self) -> dict:
        """Serialize the full tree."""
        if self.root is None:
            return {}
        return self.root.to_dict()

    def save(self, path: Path):
        """Write timing tree to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# --- Metadata collection ---


def _run_git(*args: str) -> str:
    """Run a git command and return stripped stdout, or '' on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def collect_metadata() -> dict:
    """Collect environment metadata for a benchmark run.

    Returns
    -------
    dict
        Git info, hardware, Python/torch versions, hostname.
    """
    git_hash = _run_git("rev-parse", "--short", "HEAD")
    git_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    git_dirty = _run_git("status", "--porcelain") != ""

    gpu_name = ""
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "git_hash": git_hash,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "platform": platform.platform(),
    }
