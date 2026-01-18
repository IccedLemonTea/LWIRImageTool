### CalibrationData Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 11/08/2025
# File : CalibrationData.py

from pydantic import BaseModel, ConfigDict
import numpy as np
from typing import Optional

class CalibrationData(BaseModel):
    """
    Base calibration data container.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_stack: Optional[np.ndarray] = None
    coefficients: Optional[np.ndarray] = None

