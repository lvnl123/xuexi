from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel,QPushButton


class Mywindow(QWidget):
    def __init__(self):
        super().__init__()
        self.mainLayout = QVBoxLayout()
        self.label = QLabel('主窗口')
        self.subwindow = Subwindow()
        self.btn_show = QPushButton('显示子窗口')
        self.btn_close = QPushButton('关闭子窗口')
        self.btn_hide = QPushButton('隐藏子窗口')
        self.btn_show.clicked.connect(self.openSubwindow)
        self.btn_close.clicked.connect(self.closeSubwindow)
        self.btn_hide.clicked.connect(self.hideSubwindow)
        self.mainLayout.addWidget(self.btn_hide)
        self.mainLayout.addWidget(self.btn_close)
        self.mainLayout.addWidget(self.btn_show)

        self.setLayout(self.mainLayout)

    def openSubwindow(self):
        self.subwindow.show()

    def closeSubwindow(self):
        self.subwindow.close()

    def hideSubwindow(self):
        self.subwindow.hide()

class Subwindow(QWidget):
     def __init__(self):
        super().__init__()
        self.mainLayout = QVBoxLayout()
        self.label = QLabel("子窗口")
        self.mainLayout.addWidget(self.label)
        self.setLayout(self.mainLayout)

if __name__ == '__main__':
        app = QApplication([])
        window = Mywindow()
        window.show()
        app.exec()