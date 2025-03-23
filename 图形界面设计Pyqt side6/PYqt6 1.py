import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit

app = QApplication([])
window = QMainWindow()
window.resize(500, 400)
window.move(100, 200)
window.setWindowTitle('My Window')

textEdit = QPlainTextEdit(window)
textEdit.setPlaceholderText('Your Name')
textEdit.move(10, 25)

window.show()
app.exec()