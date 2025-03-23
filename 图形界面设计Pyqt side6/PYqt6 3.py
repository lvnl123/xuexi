from PySide6.QtWidgets import QPushButton, QApplication
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

mainwin_wide = 600
mainwin_height = 480


class Mainwindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("身材管理系统")
        self.resize(mainwin_wide, mainwin_height)

        self.label_dark = QtWidgets.QLabel(self)
        self.label_dark.resize(100, 100)
        pic = QtGui.QPicture()
        painter = QtGui.QPainter(pic)
        painter.fillRect(0, 0, 100, 100, QtGui.QColor(128, 128, 128))
        painter.end()
        self.label_dark.setPicture(pic)  # 用于展示QPicture
        self.label_dark.move(25, 25)

        self.label_head = QtWidgets.QLabel(self)
        self.label_head.move(20, 20)
        self.label_head.setPixmap(QtGui.QPixmap('1055B2D0E4E696918CF810449626AD65.png'))

        self.label_line1 = QtWidgets.QLabel(self)
        self.label_line1.move(10, 130)
        self.label_line1.setText("-----------------------------------------------------------------------------------"
                                 "-----------------------------")

        self.label_tip = QtWidgets.QLabel(self)
        self.label_tip.move(150, 20)
        self.label_tip.setTextFormat(Qt.TextFormat.MarkdownText)
        self.label_tip.setText("### 请选择餐别：")

        self.cbb_type = QtWidgets.QComboBox(self)
        self.cbb_type.move(240, 17)
        self.cbb_type.addItem("早餐")
        self.cbb_type.addItem("午餐")
        self.cbb_type.addItem("晚餐")
        self.cbb_type.resize(135, 25)

        self.label_tip2 = QtWidgets.QLabel(self)
        self.label_tip2.move(150, 60)
        self.label_tip2.setTextFormat(Qt.TextFormat.MarkdownText)
        self.label_tip2.setText("### 请输入食物：")

        self.line_edit_food = QtWidgets.QLineEdit(self)
        self.line_edit_food.move(240, 60)
        self.label_tip3 = QtWidgets.QLabel(self)
        self.label_tip3.move(150, 100)
        self.label_tip3.setTextFormat(Qt.TextFormat.MarkdownText)
        self.label_tip3.setText("### 请输入数量：")

        self.line_edit_que = QtWidgets.QLineEdit(self)
        self.line_edit_que.move(240, 100)

        self.label_tip4 = QtWidgets.QLabel(self)
        self.label_tip4.move(380, 100)
        self.label_tip4.setTextFormat(Qt.TextFormat.MarkdownText)
        self.label_tip4.setText("（ 单位：克 ）")

        self.btn_add = QPushButton('\n 录  入      \n', self)
        self.btn_add.move(490, 65)


if __name__ == '__main__':
    app = QApplication([])
    window = Mainwindow()
    window.show()
    app.exec()