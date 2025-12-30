### RJPEG Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 10/27/2025
# File : RJPEG.py


import LWIRimagetool

import os
import numpy as np
import subprocess
import PIL
import io

class RJPEG(LWIRimagetool.ImageData):
    def __init__(self,filename: str):
        LWIRimagetool.ImageData.__init__(self)
        self.__reader(filename)


    def __reader(self,filename: str):
        """
        Reads the Thermal Images from RJPEG Files
        Parameters:
            filepath(str): Path to the RJPEG file
        Returns:
        np.ndarray: Array containing the Thermal Images

        """

        ## Reading in image
        cmd = ["exiftool", "-b", "-RawThermalImage", filename]
        result = subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True)
        blob = result.stdout

        raw = PIL.Image.open(io.BytesIO(blob))
        img = np.array(raw)
        if img.dtype != np.uint16:
            img = img.astype(np.uint16, copy=False)
        
        self._raw_counts = img

