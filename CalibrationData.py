### CalibrationData Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 11/08/2025
# File : CalibrationData.py


import numpy

class CalibrationData(object):

    def __init__(self):
        self._coefficients = None
        self._image_stack = None

    @property
    def coefficients(self):
        return self._coefficients
    
    @property
    def image_stack(self):
        return self._image_stack

    @image_stack.setter
    def coefficients(self, image_stack):
        self._image_stack = image_stack

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients
