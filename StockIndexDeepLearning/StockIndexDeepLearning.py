from TFMainForm_handler import *

app = QApplication(sys.argv)
mainWindow = TFMainForm_handler()
mainWindow.show()
app.exec_()