
from pydantic import BaseModel, Field, field_validator
import scipy.constants as const
import scipy.integrate as integrate
import math
import numpy as np
from typing import Sequence, Optional


class Blackbody(BaseModel):
    """
   title::
      Blackbody

   description::
      A blackbody class designed to calculate accurate radiance
      calculations necessary for readiometric calibrations of 
      detectors.

   attributes::
      None

   methods::

   author::
      Cooper White

   copyright::

   license::
      MIT
   version::
      1.0.0

   disclaimer::

      Permission is hereby granted, free of charge, to any person obtaining a copy
      of this software and associated documentation files (the "Software"), to deal
      in the Software without restriction, including without limitation the rights
      to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
      copies of the Software, and to permit persons to whom the Software is
      furnished to do so, subject to the following conditions:

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
      IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
      FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
      LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
      OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
      SOFTWARE.
   """

    absolute_temperature: float = Field(
        default=300.0,
        gt=0.0,
        description="Absolute temperature in Kelvin"
    )

    def total_radiance(self) -> float:
        """
        Computes the total hemispherical radiance of a blackbody using the
        Stefan-Boltzmann law.

        Returns
        -------
        total_radiance : float
            Total blackbody radiance in units of W/m^2/sr.
        """
        stefan_boltzmann_constant = const.sigma  # [W/m^2/K^4]

        total_radiance = stefan_boltzmann_constant * self.absolute_temperature * self.absolute_temperature * self.absolute_temperature * self.absolute_temperature

        return total_radiance  # [W/m^2/sr]

    def spectral_radiance(
        self,
        wavelengths: Sequence[float],
        rsr: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        """
        Computes the spectral radiance of a blackbody using Planck’s law.

        Radiance is evaluated at each wavelength and optionally weighted
        by a relative spectral response (RSR) function.

        Parameters
        ----------
        wavelengths : list or np.ndarray
            Wavelength values in microns.
        rsr : list or np.ndarray, optional
            Relative spectral response of the sensor system. Must be the
            same length as `wavelengths`.

        Returns
        -------
        spectral_radiance : np.ndarray
            Spectral radiance in units of W/m^2/sr/micron.
        """

        wavelengths = np.asarray(wavelengths, dtype=float)
        if rsr is not None:
            rsr = np.asarray(rsr, dtype=float)

        plancks_constant = const.h  # 6.62607015e-34 [Joules*Seconds]
        speed_of_light_constant = const.c  # 299792458.0 [Meters/Second]
        boltzmann_constant = const.k  # 1.380649e-23 [Joules/Kelvin]

        wavelength = wavelengths * 0.000001  # Microns --> Meters
        spectral_radiance = []

        numerator = (2.0 * plancks_constant * speed_of_light_constant *
                     speed_of_light_constant)

        for i in range(len(wavelength)):

            denominator = (wavelength[i] * wavelength[i] * wavelength[i] *
                           wavelength[i] * wavelength[i] * 1000000.0)
            exponent = plancks_constant * speed_of_light_constant / (
                wavelength[i] * boltzmann_constant *
                self.absolute_temperature)
            if rsr is not None:
                spectral_radiance.append(numerator / denominator * 1 /
                                         (math.exp(exponent) - 1) * rsr[i])
            else:
                spectral_radiance.append(
                    numerator / denominator * 1 /
                    (math.exp(exponent) - 1))  # [W/m^2/sr/micron]

        return spectral_radiance  # [W/m^2/sr/micron]

    def band_radiance(
        self,
        wavelengths: Sequence[float],
        rsr: Optional[Sequence[float]] = None
    ) -> float:
        """
        Computes the integrated band radiance over a wavelength interval.

        The spectral radiance is numerically integrated across the provided
        wavelength range. If an RSR is supplied, the result is normalized
        by the integrated response.

        Parameters
        ----------
        wavelengths : list or np.ndarray
            Wavelength values in microns.
        rsr : list or np.ndarray, optional
            Relative spectral response of the sensor system.

        Returns
        -------
        band_radiance : float
            Integrated band radiance in units of W/m^2/sr.
        """

        wavelengths = np.asarray(wavelengths, dtype=float)

        spectral_radiance = self.spectral_radiance(wavelengths, rsr)

        int_radiance = integrate.simpson(spectral_radiance, wavelengths)

        if rsr is not None:
            rsr = integrate.simpson(rsr, wavelengths)
            int_radiance = int_radiance/rsr # Divide by RSR to normalize and retain units of /micron
            
        return int_radiance  # [W/m^2/sr/micron]
    
if __name__ == "__main__":
    from .Blackbody import Blackbody
    import numpy as np


    rsr = "/home/cjw9009/Desktop/suas_data/flir_boson_with_13mm_45fov.txt"
    txt_content = np.loadtxt(rsr, skiprows=1, delimiter=',')
    wavelengths = txt_content[:, 0]
    response = txt_content[:, 1]

    BB = Blackbody()
    BB.absolute_temperature = 283.15

    bb_band_rad = BB.band_radiance(wavelengths)
    print(f"The total blackbody band radiance is {bb_band_rad}")
    band_rad = BB.band_radiance(wavelengths, response)
    print(f"The integrated band radiance on the sensor is {band_rad}")
