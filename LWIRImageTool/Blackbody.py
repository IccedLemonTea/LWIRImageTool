### Blackbody Class ###
# Author : Cooper White
# Date : 11/17/2025
# File : Blackbody.py

from pydantic import BaseModel, Field, field_validator
import scipy.constants as const
import scipy.integrate as integrate
import math
import numpy as np
from typing import Sequence, Optional


class Blackbody(BaseModel):

    absolute_temperature: float = Field(
        default=300.0,
        gt=0.0,
        description="Absolute temperature in Kelvin"
    )

    def total_radiance(self) -> float:
        """ 
        Calculates the total radiance of a blackbody with the absolute temperature set in the object
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
        Computes the spectral radiance of the blackbody object given an interval of wavelengths to compute over
            wavelengths(list or 1D-np.array) must be in microns
            rsr(list or 1D-np.array) relative spectral response of the system
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
        Computes the integrated band radiance of the blackbody object given an interval of wavelengths to compute over
            wavelengths(list or 1D-np.array) must be in microns
            rsr(list or 1D-np.array) relative spectral response of the system
        """

        wavelengths = np.asarray(wavelengths, dtype=float)

        spectral_radiance = self.spectral_radiance(wavelengths, rsr)

        int_radiance = integrate.simpson(spectral_radiance, wavelengths)

        if rsr is not None:
            rsr = integrate.simpson(rsr, wavelengths)
            int_radiance = int_radiance/rsr
            
        return int_radiance  # [W/m^2/sr]
    
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
