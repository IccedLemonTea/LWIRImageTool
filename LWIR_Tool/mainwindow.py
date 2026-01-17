# This Python file uses the following encoding: utf-8
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal, QThread
sys.path.append(os.path.expanduser("/home/cjw9009/Desktop/Senior_Project"))
import LWIRImageTool as lit
from CalibrationWorker import CalibrationWorker
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connecting open image tab to trigger
        self.ui.actionOpenImage.triggered.connect(self.open_image)
        # Connecting Open Directory to trigger
        self.ui.actionOpenBBDirectory.triggered.connect(self.open_blackbody_directory)

    
    # Open a directory of images and store them within the GUI's memory
    # Input : self
    # Action : Asks user to select a dir path, then loads in all imagery within that dir.
    def open_blackbody_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            ""
            )

        if not directory:
            return  # User cancelled

        QMessageBox.information(
            self,
            "Directory Selected",
            f"You selected:\n{directory}"
            )

        rsr_path = QFileDialog.getOpenFileName(
        self,
        "Select RSR file path",
        "","*.txt;;All Files (*)"
        )

        if not rsr_path:
            return  # User cancelled

        QMessageBox.information(
            self,
            "RSR Selected",
            f"You selected:\n{rsr_path}"
            )
        self.start_calibration(directory, filetype = 'rjpeg', rsr=None)

    # Open a singular image to view
    # Input : self
    # Action : Changes the main window and dispays the image that is selected.
    def open_image(self):
        print("Hello World!")

    def update_progress(self, phase, current, total):
        percent = int((current / total) * 100)

        if phase == "loading":
            self.ui.progressBar.setFormat("Loading images... %p%")
        elif phase == "calibrating":
            self.ui.progressBar.setFormat("Calibrating pixels... %p%")
        elif phase == "ascension":
            self.ui.progressBar.setFormat("Calculating ascension regions... %p%")

        self.ui.progressBar.setValue(percent)

    def calibration_finished(self, cal_array):
        self.ui.progressBar.setValue(100)
        self.ui.progressBar.setFormat("Calibration complete")

        self.calibration_data = cal_array

        QMessageBox.information(
            self,
            "Calibration Finished",
            "Blackbody calibration completed successfully."
        )
        print(f"{self.calibration_data}")

    def start_calibration(self, directory, filetype, rsr=None):
        self.thread = QThread()
        self.worker = CalibrationWorker(directory, filetype, rsr)

        self.worker.moveToThread(self.thread)

        # Wiring
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.calibration_finished)


        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # UI state
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setFormat("Starting calibration...")

        self.thread.start()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
