from PySide import QtGui
from ImageViewer import ImageViewer
import sys

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
