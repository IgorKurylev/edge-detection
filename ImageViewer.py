from PySide import QtCore, QtGui
from tools import image_pipeline
from lane import Lane
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class ImageViewer(QtGui.QWidget):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.scaleFactor = 0.0
        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)

        self.button = QtGui.QPushButton("Load Image")
        self.button.clicked.connect(self.open)

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.scrollArea)
        mainLayout.addWidget(self.button)
        self.setLayout(mainLayout)

        self.setWindowTitle("Edge detection")
        self.resize(960, 640)

    def open(self):
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                QtCore.QDir.currentPath())
        if fileName:

            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            Lane.purge()
            image = mpimg.imread(fileName)
            try:
                res = image_pipeline(image)
            except:
                QtGui.QMessageBox.information(self, "Image Viewer",
                                              "Cannot detect lines in %s." % fileName)
                return
            plt.imsave('res.jpg', res)
            image = QtGui.QImage('res.jpg')

            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
            self.scaleFactor = 1.0
            self.imageLabel.adjustSize()
