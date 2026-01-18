### ImageData Class ###
# Author : Cooper White 
# Date : 09/30/2025
# File : ImageData.py


from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from typing import Optional
class ImageData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_counts: Optional[np.ndarray] = None 
    metadata: dict = Field(default_factory=lambda: {
        'sensorType': 'Unknown',
        'bitDepth': None,
        'horizontalRes': None,
        'verticalRes': None,
        'bands': None,
        'acquisitionTime': None
    })

    def display_metadata(self):
      print('METADATA:')
      print('Sensor type: {0}'.format(
               self._metadata['sensorType']))
      print('Bit depth: {0}'.format(
               self._metadata['bitDepth']))
      print('Number of bands: {0}'.format(
               self._metadata['bands']))