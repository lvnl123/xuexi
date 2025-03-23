import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QFrame, QColorDialog
from PySide6.QtGui import QColor


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Initialize a default black color
        col = QColor(0, 0, 0)

        # Create a button
        self.btn = QPushButton('Dialog', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.showDialog)

        # Create a frame with a default black background
        self.frm = QFrame(self)
        self.frm.setStyleSheet(f"QWidget {{ background-color: {col.name()} }}")
        self.frm.setGeometry(130, 22, 100, 100)

        # Set window geometry and title
        self.setGeometry(300, 300, 250, 180)
        self.setWindowTitle('Color Dialog')
        self.show()

    def showDialog(self):
        # Open a color dialog and get the selected color
        col = QColorDialog.getColor()

        # If the selected color is valid, update the frame's background color
        if col.isValid():
            self.frm.setStyleSheet(f"QWidget {{ background-color: {col.name()} }}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())