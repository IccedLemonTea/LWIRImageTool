### ImageData Class ###
# Author : Cooper White 
# Date : 09/30/2025
# File : ImageData.py

import numpy

class ImageData(object):

    def __init__(self):
        self._raw_counts = None
        self._metadata = {'sensorType': 'Unknown',
                        'bitDepth': None,
                        'horizontalRes' : None,
                        'verticalRes' : None,
                        'bands' : None,
                        'acquisitionTime' : None
                        }
    
    
    @property
    def raw_counts(self):
        return self._raw_counts

    @raw_counts.setter
    def raw_counts(self, raw_counts):
        self._raw_counts = raw_counts

    @raw_counts.deleter
    def raw_counts(self):
        del self._raw_counts

    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata):
        self._metadata.update({'sensorType': metadata['sensorType']})
        self._metadata.update({'bitDepth': metadata['bitDepth']})
        self._metadata.update({'bands': metadata['bands']})

    @metadata.deleter
    def metadata(self):
      del self._metadata

    def display_metadata(self):
      print('METADATA:')
      print('Sensor type: {0}'.format(
               self._metadata['sensorType']))
      print('Bit depth: {0}'.format(
               self._metadata['bitDepth']))
      print('Number of bands: {0}'.format(
               self._metadata['bands']))