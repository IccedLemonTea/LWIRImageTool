### BlackbodyCalibrationConfig Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 01/14/2026
# File : BlackbodyCalibrationConfig.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Callable, Union
import numpy as np


class BlackbodyCalibrationConfig(BaseModel):
    """
    Configuration for a blackbody calibration run.

    Pass an instance of this class to ``CalibrationDataFactory.create()``
    to produce a ``BlackbodyCalibration`` object.

    Parameters
    ----------
    directory : str
        Path to the directory containing the blackbody image sequence.
    filetype : str, optional
        Image file format.  Supported values: ``'rjpeg'``, ``'envi'``.
        Default is ``'rjpeg'``.
    blackbody_temperature : float
        Starting blackbody temperature in Kelvin.  Must be > 0.
    temperature_step : float
        Temperature increment between successive blackbody steps in Kelvin.
        Must be > 0.
    rsr : str, np.ndarray, or None, optional
        Relative spectral response definition.

        * ``str``  — path to a two-column ``.txt`` file (wavelength [µm],
          response); loaded with ``np.loadtxt(..., skiprows=1, delimiter=',')``.
        * ``np.ndarray`` — shape ``(2, N)`` where ``[0, :]`` is wavelength
          [µm] and ``[1, :]`` is the normalised response.
        * ``None`` — flat response from 8-14 µm assumed.

        Default is ``None``.
    progress_cb : callable or None, optional
        Callback for progress updates.  Called as
        ``progress_cb(phase, current, total)`` where *phase* is a string
        label (e.g. ``'loading'``, ``'calibrating'``), *current* and *total*
        are integers.  Default is ``None``.
    chunk_fraction : float, optional
        Fraction of the signal length used for chunk averaging.
        Must be in ``(0, 1]``.  Default is ``0.01``.
    deriv_threshold : float, optional
        Standard-deviation multiplier used when detecting derivative peaks
        during ascension detection.  Default is ``3``.
    window_fraction : float, optional
        Fraction of the total frame count used as the search window when
        matching derivative peaks to temperature steps.  Default is ``0.001``.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    directory: str
    filetype: str = "rjpeg"

    blackbody_temperature: float = Field(
        ..., gt=0, description="Starting temperature [K]")
    temperature_step: float = Field(...,
                                    gt=0,
                                    description="Temperature step [K]")

    rsr: Optional[Union[str, np.ndarray]] = None

    progress_cb: Optional[Callable] = None

    chunk_fraction: float = Field(default=0.01, gt=0, le=1)
    deriv_threshold: float = Field(default=3, gt=0)
    window_fraction: float = Field(default=0.001, gt=0, le=1)
