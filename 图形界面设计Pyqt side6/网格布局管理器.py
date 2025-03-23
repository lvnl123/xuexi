from PySide6 import QtWidgets  # Import QtWidgets for GUI components


class Example(QtWidgets.QWidget):  # Define the class and inherit from QWidget
    def __init__(self):
        super().__init__()  # Correctly call the parent class constructor
        self.initUI()  # Call the initUI method to set up the UI

    def initUI(self):
        # List of button names
        names = ['Cls', 'Bck', '', 'Close',
                 '7', '8', '9', '/',
                 '4', '5', '6', '*',
                 '1', '2', '3', '-',
                 '0', '.', '=', '+']

        # Create a grid layout
        grid = QtWidgets.QGridLayout()

        # Button positions in the grid
        pos = [(i, j) for i in range(5) for j in range(4)]

        # Add buttons to the grid
        for name, position in zip(names, pos):
            if name == '':
                # Add an empty label for the empty cell
                grid.addWidget(QtWidgets.QLabel(''), position[0], position[1])
            else:
                # Create and add a button
                button = QtWidgets.QPushButton(name)
                grid.addWidget(button, position[0], position[1])

        # Set the layout for the main window
        self.setLayout(grid)

        # Set the geometry and title of the window
        self.move(300, 150)
        self.setWindowTitle('我的计算器')

        # Show the window
        self.show()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)  # Create the application object
    ex = Example()  # Create an instance of the Example class
    sys.exit(app.exec())  # Start the event loop (use exec() for PySide6)