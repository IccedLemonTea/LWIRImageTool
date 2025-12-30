### ENVI Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 09/30/2025
# File : ENVI.py


import LWIRimagetool

import os
import numpy as np
import spectral.io.envi as envi

class ENVI(LWIRimagetool.ImageData):
    def __init__(self,filename):
        LWIRimagetool.ImageData.__init__(self)
        self.__reader(filename)


    def __reader(self,filename):
        """
        Reads the Thermal Images from ENVI Files
        Parameters:
            filepath(str): Path to the ENVI file
            validate(bool): Whether to validate the file
        Returns:
        np.ndarray: Array containing the Thermal Images

        """

        ## Reading in image
        image = envi.open(filename + ".hdr",filename)
        self._raw_counts = image.load()

        ## Updating metadata
        self._metadata.update({'bands': image.metadata['bands']})
        self._metadata.update({'sensorType': 'ENVI'})
        self._metadata.update({'bitDepth': ENVI.envi_dtype_to_bitdepth(image.metadata['data type'])})
        self._metadata.update({'horizontalRes' : image.metadata['samples']})
        self._metadata.update({'verticalRes' : image.metadata['lines']})

    def envi_dtype_to_bitdepth(dtype_code: str) -> int:
        """
        Accepts ENVI data type code as a string and returns
        the corresponding bit depth (int). Returns None if not found.
        """
        mapping = {
            '1': 8,    # Byte: 8-bit unsigned int
            '2': 16,   # Integer: 16-bit signed int
            '3': 32,   # Long Integer: 32-bit signed int
            '4': 32,   # Float: 32-bit float
            '5': 64,   # Double: 64-bit double
            '6': 64,   # Complex: 2×32-bit floats = 64 bits total
            '9': 128,  # Double-precision Complex: 2×64-bit = 128 bits total
            '12': 16,  # Unsigned Integer: 16-bit unsigned
            '13': 32,  # Unsigned Long Integer: 32-bit unsigned
            '14': 64,  # 64-bit signed integer
            '15': 64,  # 64-bit unsigned integer
        }
        return mapping.get(dtype_code, None)
