# qtpy defaults to PyQt5/PySide2 which can be present in upgraded environments
import qtpy

if qtpy.PYQT5:
    raise RuntimeError(
        "Please remove PyQt5 from your environment with `pip uninstall PyQt5`"
    )
elif qtpy.PYSIDE2:
    raise RuntimeError(
        "Please remove PySide2 from your environment with `pip uninstall PySide2`"
    )
