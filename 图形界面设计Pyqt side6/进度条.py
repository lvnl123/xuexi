import sys
from PySide6.QtWidgets import QApplication, QWidget, QProgressBar, QPushButton
from PySide6.QtCore import QBasicTimer, Qt


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):
        self.pbar = QProgressBar(self)  # 创建进度条
        self.pbar.setGeometry(30, 40, 200, 25)  # 设置进度条的位置和大小

        self.btn = QPushButton('Start', self)  # 创建按钮
        self.btn.move(40, 80)  # 设置按钮的位置
        self.btn.clicked.connect(self.doAction)  # 连接按钮点击事件到槽函数

        self.timer = QBasicTimer()  # 创建定时器
        self.step = 0  # 初始化进度值

        self.setGeometry(300, 300, 280, 170)  # 设置窗口的位置和大小
        self.setWindowTitle('QtGui.QProgressBar')  # 设置窗口标题
        self.show()  # 显示窗口

    def timerEvent(self, e):
        if self.step >= 100:
            self.timer.stop()  # 停止定时器
            self.btn.setText('Finished')  # 更新按钮文本为"Finished"
            return
        self.step = self.step + 1  # 增加进度值
        self.pbar.setValue(self.step)  # 更新进度条的值

    def doAction(self):
        if self.timer.isActive():
            self.timer.stop()  # 如果定时器正在运行，则停止它
            self.btn.setText('Start')  # 更新按钮文本为"Start"
        else:
            self.timer.start(100, self)  # 启动定时器，每100毫秒触发一次timerEvent
            self.btn.setText('Stop')  # 更新按钮文本为"Stop"


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()