# qtpy defaults to PyQt5/PySide2 which can be present in upgraded environments
try:
    import qtpy

    qtpy.API_NAME  # check qtpy API name - one is required for GUI

except RuntimeError as error:
    if type(error).__name__ == "QtBindingsNotFoundError":
        print("WARNING: QtBindings (PyQT or PySide) was not found for GUI")
