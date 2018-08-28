import sys
from MainForm_handler import *

app = QApplication(sys.argv)
mainWindow = MainForm_handler()
mainWindow.show()
app.exec_()