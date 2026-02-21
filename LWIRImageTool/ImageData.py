
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from typing import Optional


class ImageData(BaseModel):
   """
    Container for a single LWIR image and its metadata.

    Attributes
    ----------
    raw_counts : np.ndarray or None
        2-D array of raw digital counts, shape ``(rows, cols)``.
        ``None`` until populated by a reader subclass.
    metadata : dict
        Sensor and acquisition metadata with the following keys:

        * ``'sensorType'``    — sensor identifier string.
        * ``'bitDepth'``      — integer bit depth of the sensor.
        * ``'horizontalRes'`` — number of columns.
        * ``'verticalRes'``   — number of rows.
        * ``'bands'``         — number of spectral bands.
        * ``'acquisitionTime'`` — acquisition timestamp or ``None``.
    """
   model_config = ConfigDict(arbitrary_types_allowed=True)

   raw_counts: Optional[np.ndarray] = None
   metadata: dict = Field(
      default_factory=lambda: {
         'sensorType': 'Unknown',
         'bitDepth': None,
         'horizontalRes': None,
         'verticalRes': None,
         'bands': None,
         'acquisitionTime': None
      })

   def display_metadata(self):
      print('METADATA:')
      print('Sensor type: {0}'.format(self._metadata['sensorType']))
      print('Bit depth: {0}'.format(self._metadata['bitDepth']))
      print('Number of bands: {0}'.format(self._metadata['bands']))
