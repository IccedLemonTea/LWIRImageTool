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
    blackbody_temperature : int or None
        Starting blackbody temperature [K] used during the calibration run.
    temperature_step : int or None
        Temperature increment between successive steps [K].
    rsr : str or None
        Path to the RSR file used, or ``None`` if a simulated or default
        RSR was applied.
    deriv_threshold : float or None
        Standard-deviation multiplier used for ascension detection.
    window_fraction : float or None
        Frame-fraction search window used for ascension detection.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_stack: Optional[np.ndarray] = None
    coefficients: Optional[np.ndarray] = None
    directory : Optional[str] = None
    blackbody_temperature: Optional[int] = None
    temperature_step: Optional[int] = None  
    rsr: Optional[str] = None
    deriv_threshold: Optional[float] = None
    window_fraction: Optional[float] = None

