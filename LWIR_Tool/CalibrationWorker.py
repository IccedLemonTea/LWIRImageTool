# This Python file uses the following encoding: utf-8
from PySide6.QtCore import QObject, Signal
import sys
import os
sys.path.append(os.path.expanduser("/home/cjw9009/Desktop/Senior_Project"))
import LWIRImageTool as lit


class CalibrationWorker(QObject):
    progress = Signal(str, int, int)   # phase, current, total
    finished = Signal(object)          # calibration array
    error = Signal(str)

    def __init__(self, directory, filetype, rsr=None):
        super().__init__()
        self.directory = directory
        self.filetype = filetype
        self.rsr = rsr

    def run(self):
        factory = lit.CalibrationDataFactory()

        config = lit.BlackbodyCalibrationConfig(
        directory = self.directory,
        filetype = self.filetype,
        blackbody_temperature = 283.15,
        temperature_step = 5.0,
        rsr = self.rsr,
        progress_cb = self._progress_callback)

        result = factory.create(config)

        self.finished.emit(result)


    def _progress_callback(self, phase, current, total):
        self.progress.emit(phase, current, total)
