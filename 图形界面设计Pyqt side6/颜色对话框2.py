from PySide6 import QtWidgets  # Import QtWidgets for GUI components


class Example(QtWidgets.QWidget):  # Define the class and inherit from QWidget
    def __init__(self):
        super().__init__()  # Correctly call the parent class constructor
        self.initUI()  # Call the initUI method to set up the UI

    def initUI(self):
        # Create a vertical layout
        vbox = QtWidgets.QVBoxLayout()

        # Create a button
        btn = QtWidgets.QPushButton('Open Color Dialog', self)
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,  # Use Policy.Fixed for PySide6
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        # Add the button to the layout
        vbox.addWidget(btn)

        # Connect the button's clicked signal to the showDialog method
        btn.clicked.connect(self.showDialog)

        # Create a frame to display the selected color
        self.color_frame = QtWidgets.QFrame(self)
        self.color_frame.setStyleSheet("background-color: white;")
        self.color_frame.setFixedSize(100, 100)

        # Add the frame to the layout
        vbox.addWidget(self.color_frame)

        # Set the layout for the main window
        self.setLayout(vbox)

        # Set the geometry and title of the window
        self.setGeometry(300, 300, 250, 200)
        self.setWindowTitle('Color Dialog')

        # Show the window
        self.show()

    def showDialog(self):
        # Open a color dialog and get the selected color
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():  # Check if a valid color was selected
            # Update the frame's background color
            self.color_frame.setStyleSheet(f"background-color: {color.name()};")
            print(f"Selected color: {color.name()}")


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)  # Create the application object
    ex = Example()  # Create an instance of the Example class
    sys.exit(app.exec())  # Start the event loop (use exec() for PySide6)