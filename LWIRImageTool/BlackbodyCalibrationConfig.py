### BlackbodyCalibrationConfig Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 01/14/2026
# File : BlackbodyCalibrationConfig.py

from pydantic import BaseModel, Field
from typing import Optional, Callable


class BlackbodyCalibrationConfig(BaseModel):
    """
    Object for setting all necessary values in a blackbody calibration run

    Variables
    ----------
    directory : str
        Path to the directory containing blackbody images.
    filetype : str
        Type/format of the image files (e.g., 'rjpeg', 'envi').
    blackbody_temperature : float
        Temperature that blackbody starts the run at in [K]
    temperature_step : float
        Temperature value that the blackbody changes by between each step in [K]
    rsr : str or None
        Path to the RSR file containing spectral response and wavelengths. If None, 
        default wavelengths are used. (8-14 microns)
    progress_cb : callable, optional
        Callback function for progress updates. Called with `phase`, `current`, and `total`.
    deriv_threshold : int
        Factor to distinguish how far from the stdev the derivative being checked is. Default is 3
    window_fraction : float
        Fraction of data to be searched when finding ascensions. Default is 0.001
    """
    directory: str
    filetype: str = "rjpeg"

    blackbody_temperature: float = Field(
        ..., gt=0, description="Starting temperature [K]")
    temperature_step: float = Field(...,
                                    gt=0,
                                    description="Temperature step [K]")

    rsr: Optional[str] = None
    progress_cb: Optional[Callable] = None

    chunk_fraction: float = Field(default=0.01, gt=0, le=1)
    deriv_threshold: float = Field(default=3, gt=0)
    window_fraction: float = Field(default=0.001, gt=0, le=1)
