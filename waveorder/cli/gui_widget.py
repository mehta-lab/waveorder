import sys

import click

try:
    from waveorder.plugin import tab_recon
except:
    pass

try:
    from qtpy.QtWidgets import QApplication, QStyle, QVBoxLayout, QWidget
except:
    pass

try:
    import qdarktheme  # pip install pyqtdarktheme==2.1.0 --ignore-requires-python
except:
    pass

PLUGIN_NAME = "waveorder: Computational Toolkit for Label-Free Imaging"
PLUGIN_ICON = "🔬"


@click.command()
def gui():
    """GUI for waveorder: Computational Toolkit for Label-Free Imaging"""

    app = QApplication(sys.argv)
    app.setStyle(
        "Fusion"
    )  # Other options: "Fusion", "Windows", "macOS", "WindowsVista"
    try:
        qdarktheme.setup_theme("dark")
    except Exception as e:
        print(e.args)
        pass
    window = MainWindow()
    window.setWindowTitle(PLUGIN_ICON + " " + PLUGIN_NAME + " " + PLUGIN_ICON)

    pixmapi = getattr(QStyle.StandardPixmap, "SP_TitleBarMenuButton")
    icon = app.style().standardIcon(pixmapi)
    window.setWindowIcon(icon)

    window.show()
    sys.exit(app.exec())


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        recon_tab = tab_recon.Ui_ReconTab_Form(stand_alone=True)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(recon_tab.recon_tab_mainScrollArea)


if __name__ == "__main__":
    gui()
