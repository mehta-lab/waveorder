import sys
import click

try:
    from recOrder.plugin import tab_recon
except:pass

try:
    from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QStyle
except:pass

try:
    import qdarktheme
except:pass

PLUGIN_NAME = "recOrder: Computational Toolkit for Label-Free Imaging"
PLUGIN_ICON = "🔬"

@click.command()
def gui():
    """GUI for recOrder: Computational Toolkit for Label-Free Imaging"""

    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Other options: "Fusion", "Windows", "macOS", "WindowsVista"
    try:
        qdarktheme.setup_theme("dark")
    except:pass
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
