import sys
from PySide6.QtWidgets import QApplication, QWidget, QSlider, QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):
        sld = QSlider(Qt.Horizontal, self)  # 创建水平滑块
        sld.setFocusPolicy(Qt.NoFocus)  # 设置焦点策略为无焦点
        sld.setGeometry(30, 40, 100, 30)  # 设置滑块的位置和大小
        sld.valueChanged[int].connect(self.changeValue)  # 连接滑块值变化信号到槽函数

        self.label = QLabel(self)  # 创建标签
        self.label.setPixmap(QPixmap('mute.png'))  # 设置初始图标为'mute.png'
        self.label.setGeometry(160, 40, 80, 30)  # 设置标签的位置和大小

        self.setGeometry(300, 300, 280, 170)  # 设置窗口的位置和大小
        self.setWindowTitle('QtGui.QSlider')  # 设置窗口标题
        self.show()  # 显示窗口

    def changeValue(self, value):
        if value == 0:
            self.label.setPixmap(QPixmap('mute.png'))  # 当滑块值为0时，设置图标为'mute.png'
        elif value > 0 and value <= 30:
            self.label.setPixmap(QPixmap('min.png'))  # 当滑块值在1-30之间时，设置图标为'min.png'
        elif value > 30 and value < 80:
            self.label.setPixmap(QPixmap('med.png'))  # 当滑块值在31-79之间时，设置图标为'med.png'
        else:
            self.label.setPixmap(QPixmap('max.png'))  # 当滑块值大于等于80时，设置图标为'max.png'


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()