from PySide6 import QtWidgets  # Import QtWidgets for GUI components

class Example(QtWidgets.QWidget):  # Define the class and inherit from QWidget
    def __init__(self):
        super().__init__()  # Correctly call the parent class constructor
        self.initUI()  # Call the initUI method to set up the UI

    def initUI(self):
        # Create buttons
        okButton = QtWidgets.QPushButton("OK")
        cancelButton = QtWidgets.QPushButton("Cancel")

        # Create a horizontal layout and add buttons
        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)  # Add stretchable space before the buttons
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        # Create a vertical layout and add the horizontal layout
        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)  # Add stretchable space above the buttons
        vbox.addLayout(hbox)

        # Set the layout for the main window
        self.setLayout(vbox)

        # Set the geometry and title of the window
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')

        # Show the window
        self.show()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)  # Create the application object
    ex = Example()  # Create an instance of the Example class
    sys.exit(app.exec())  # Start the event loop (use exec() for PySide6)