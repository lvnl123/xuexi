import sys
from PySide6.QtWidgets import QApplication, QWidget, QCalendarWidget, QLabel
from PySide6.QtCore import QDate


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):
        cal = QCalendarWidget(self)  # 创建日历控件
        cal.setGridVisible(True)  # 显示网格线
        cal.move(20, 20)  # 设置日历的位置
        cal.clicked[QDate].connect(self.showDate)  # 连接点击事件到槽函数

        self.lbl = QLabel(self)  # 创建标签用于显示选中日期
        date = cal.selectedDate()  # 获取当前选中的日期
        self.lbl.setText(date.toString())  # 将日期转换为字符串并设置到标签中
        self.lbl.move(130, 260)  # 设置标签的位置

        self.setGeometry(300, 300, 350, 300)  # 设置窗口的位置和大小
        self.setWindowTitle('Calendar')  # 设置窗口标题
        self.show()  # 显示窗口

    def showDate(self, date):
        self.lbl.setText(date.toString())  # 更新标签文本为选中的日期


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()