from recOrder.scripts.samples import download_and_unzip


def test_download_and_unzip():
    p1, p2 = download_and_unzip("target")

    assert p1.exists()
    assert p2.exists()
