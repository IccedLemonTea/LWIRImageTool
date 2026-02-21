
from typing import Union
from .BlackbodyCalibrationConfig import BlackbodyCalibrationConfig


class CalibrationDataFactory(object):
   """
    Factory for creating source-agnostic calibration data objects.

    Dispatches to the correct ``CalibrationData`` subclass based on the
    type of config object provided.  Adding a new calibration method only
    requires adding a new ``isinstance`` branch here — callers are
    unaffected.

    Methods
    -------
    create(config)
        Construct and return a ``CalibrationData`` subclass instance.

    Examples
    --------
    >>> config = BlackbodyCalibrationConfig(
    ...     directory="/data/cal_run",
    ...     blackbody_temperature=283.15,
    ...     temperature_step=5.0,
    ... )
    >>> cal = CalibrationDataFactory.create(config)
    >>> cal.coefficients.shape
    (512, 640, 2)
    """

   @staticmethod
   def create(config: Union[BlackbodyCalibrationConfig]):  # Insert other modes of Cal here
      """
        Construct a ``CalibrationData`` object from a config.

        Parameters
        ----------
        config : BlackbodyCalibrationConfig
            A validated calibration configuration.  Additional config types
            can be added in future without changing this interface.

        Returns
        -------
        CalibrationData
            A fully populated calibration data object (currently always a
            ``BlackbodyCalibration`` instance).

        Raises
        ------
        ValueError
            If *config* is not a recognised configuration type.
       """
      from .BlackbodyCalibration import BlackbodyCalibration
      if isinstance(config, BlackbodyCalibrationConfig):
         return BlackbodyCalibration(config)

      raise ValueError(f"Unsupported calibration config: {type(config)}")


if __name__ == '__main__':
    import cv2
    import os.path
    import spectral
