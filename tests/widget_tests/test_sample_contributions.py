import urllib.error

import pytest

from waveorder.plugin.samples import download_and_unzip


def test_download_and_unzip():
    try:
        p1, p2 = download_and_unzip("target")
    except urllib.error.URLError as e:
        pytest.skip(f"sample data server unavailable: {e}")

    assert p1.exists()
    assert p2.exists()
