### RJPEG Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 10/27/2025
# File : RJPEG.py


import LWIRImageTool

import os
import numpy as np
import subprocess
import PIL.Image
import io

class RJPEG(LWIRImageTool.ImageData):
    def __init__(self,filename: str):
        LWIRImageTool.ImageData.__init__(self)
        self.__reader(filename)


    def __reader(self,filename: str):
        """
        Reads the Thermal Images from RJPEG Files
        Parameters:
            filepath(str): Path to the RJPEG file
        Returns:
        np.ndarray: Array containing the Thermal Images

        """
        if not filename.endswith("_R.jpg"):
            print(f"Skipping RJPEG file: {filename}")
            return None

        try:
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
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Exiftool failed for {filename}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to decode thermal image: {filename}") from e


