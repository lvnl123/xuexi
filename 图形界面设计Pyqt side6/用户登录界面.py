from PySide6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户登录界面")
        self.setGeometry(100, 100, 400, 250)  # 设置窗口大小和位置
        self.center_window()  # 让窗口居中显示

        # 设置字体
        font = QFont()
        font.setPointSize(10)

        # 用户名标签和文本框
        self.label_username = QLabel("用户名:", self)
        self.label_username.setFont(font)
        self.label_username.setGeometry(80, 50, 60, 30)  # x, y, 宽, 高
        self.textbox_username = QLineEdit(self)
        self.textbox_username.setFont(font)
        self.textbox_username.setGeometry(150, 50, 180, 30)

        # 密码标签和文本框
        self.label_password = QLabel("密码:", self)
        self.label_password.setFont(font)
        self.label_password.setGeometry(80, 100, 60, 30)
        self.textbox_password = QLineEdit(self)
        self.textbox_password.setFont(font)
        self.textbox_password.setEchoMode(QLineEdit.Password)  # 设置为密码模式
        self.textbox_password.setGeometry(150, 100, 180, 30)

        # 登录按钮
        self.btn_login = QPushButton("登录", self)
        self.btn_login.setFont(font)
        self.btn_login.setGeometry(100, 160, 80, 35)
        self.btn_login.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")

        # 清空按钮
        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setFont(font)
        self.btn_clear.setGeometry(220, 160, 80, 35)
        self.btn_clear.setStyleSheet("background-color: #FF5722; color: white; border-radius: 5px;")
        self.btn_clear.clicked.connect(self.clear_textboxes)

    def clear_textboxes(self):
        """清空用户名和密码文本框的内容"""
        self.textbox_username.clear()
        self.textbox_password.clear()

    def center_window(self):
        """让窗口居中显示"""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())


if __name__ == '__main__':
    app = QApplication([])
    window = LoginWindow()
    window.show()
    app.exec()