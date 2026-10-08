import sys

PLUGIN_NAME = "waveorder: Wave-optical simulation and reconstruction"
PLUGIN_ICON = "🔬"


def gui():
    """GUI for waveorder: Wave-optical simulation and reconstruction."""
    from qtpy.QtWidgets import QApplication, QStyle, QVBoxLayout, QWidget

    from waveorder.plugin import tab_recon

    class MainWindow(QWidget):
        def __init__(self):
            super().__init__()
            recon_tab = tab_recon.Ui_ReconTab_Form(stand_alone=True)
            layout = QVBoxLayout()
            self.setLayout(layout)
            layout.addWidget(recon_tab.recon_tab_mainScrollArea)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        import qdarktheme

        qdarktheme.setup_theme("dark")
    except ImportError:
        pass

    window = MainWindow()
    window.setWindowTitle(PLUGIN_ICON + " " + PLUGIN_NAME + " " + PLUGIN_ICON)
    pixmapi = getattr(QStyle.StandardPixmap, "SP_TitleBarMenuButton")
    window.setWindowIcon(app.style().standardIcon(pixmapi))
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    gui()
