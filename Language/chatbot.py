import sys
from PySide6.QtWidgets import QApplication

from UI.mainwindow import ChatBot


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatBot()
    window.show()
    sys.exit(app.exec())