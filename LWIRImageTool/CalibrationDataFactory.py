### CalibrationDataFactory Class ###
# Author : Cooper White
# Date : 11/07/2025
# File : CalibrationDataFactory.py

from typing import Union
from .BlackbodyCalibrationConfig import BlackbodyCalibrationConfig


class CalibrationDataFactory(object):
    """
   title::
      CalibrationDataFactory

   description::
      A public factory class for creating source-agnostic calibration 
      data objects. Image files from various sources will be read 
      and used to produce source-agnostic calibration data objects

   attributes::
      None

   methods::

   author::
      Cooper White

   copyright::

   license::

   version::
      1.0.0

   disclaimer::
      This source code is provided "as is" and without warranties as to 
      performance or merchantability. The author and/or distributors of 
      this source code may have made statements about this source code. 
      Any such statements do not constitute warranties and shall not be 
      relied on by the user in deciding whether to use this source code.
      
      This source code is provided without any express or implied warranties 
      whatsoever. Because of the diversity of conditions and hardware under 
      which this source code may be used, no warranty of fitness for a 
      particular purpose is offered. The user is advised to test the source 
      code thoroughly before relying on it. The user must assume the entire 
      risk of using the source code.
   """

    @staticmethod
    def create(config: Union[BlackbodyCalibrationConfig]
               ):  # Insert other modes of Cal here
        from .BlackbodyCalibration import BlackbodyCalibration
        if isinstance(config, BlackbodyCalibrationConfig):
            return BlackbodyCalibration(config)

        raise ValueError(f"Unsupported calibration config: {type(config)}")


if __name__ == '__main__':
    import cv2
    import os.path
    import spectral
