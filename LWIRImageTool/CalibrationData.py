### CalibrationData Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 11/08/2025
# File : CalibrationData.py


from pydantic import BaseModel, ConfigDict
import numpy as np
from typing import Optional


class CalibrationData(BaseModel):
    """
    Base container for calibration results and the configuration that
    produced them.

    Subclasses populate all fields during their own initialisation.
    This class imposes no calibration method — it exists to provide a
    common, type-safe data contract and to keep the producing configuration
    co-located with the results for serialisation and auditing purposes.

    Attributes
    ----------
    image_stack : np.ndarray or None
        3-D array of stacked calibration images, shape
        ``(rows, cols, frames)``.  ``None`` until populated by a subclass.
    coefficients : np.ndarray or None
        3-D array of per-pixel calibration coefficients, shape
        ``(rows, cols, 2)``, where ``[:, :, 0]`` is gain and
        ``[:, :, 1]`` is bias.  ``None`` until populated by a subclass.
    directory : str or None
        Path to the directory that was used as the image source.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_stack: Optional[np.ndarray] = None
    coefficients: Optional[np.ndarray] = None
    directory : Optional[str] = None

