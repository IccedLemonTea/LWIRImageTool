### Blackbody Class ###
# Author : Cooper White
# Date : 11/17/2025
# File : Blackbody.py

import scipy.constants as const
import scipy.integrate as integrate
import math
import numpy as np


class Blackbody(object):

    def __init__(self):
        self._absolute_temperature = 300

    @property
    def absolute_temperature(self):
        return self._absolute_temperature

    @absolute_temperature.setter
    def absolute_temperature(self, absolute_temperature: float):
        self._absolute_temperature = absolute_temperature

    def total_radiance(self) -> float:
        """ 
        Calculates the total radiance of a blackbody with the absolute temperature set in the object
        """
        stefan_boltzmann_constant = const.sigma  # [W/m^2/K^4]

        total_radiance = stefan_boltzmann_constant * self._absolute_temperature * self._absolute_temperature * self._absolute_temperature * self._absolute_temperature

        return total_radiance  # [W/m^2/sr]

    def spectral_radiance(self, wavelengths: list, rsr=None) -> list:
        """
        Computes the spectral radiance of the blackbody object given an interval of wavelengths to compute over
            wavelengths(list or 1D-np.array) must be in microns
            rsr(list or 1D-np.array) relative spectral response of the system
        """

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
                self._absolute_temperature)
            if rsr is not None:
                spectral_radiance.append(numerator / denominator * 1 /
                                         (math.exp(exponent) - 1) * rsr[i])
            else:
                spectral_radiance.append(
                    numerator / denominator * 1 /
                    (math.exp(exponent) - 1))  # [W/m^2/sr/micron]

        return spectral_radiance  # [W/m^2/sr/micron]

    def band_radiance(self, wavelengths: list, rsr=None) -> float:
        """
        Computes the integrated band radiance of the blackbody object given an interval of wavelengths to compute over
            wavelengths(list or 1D-np.array) must be in microns
            rsr(list or 1D-np.array) relative spectral response of the system
        """

        spectral_radiance = self.spectral_radiance(wavelengths, rsr)

        int_radiance = integrate.simpson(spectral_radiance, wavelengths)
        return int_radiance  # [W/m^2/sr]
