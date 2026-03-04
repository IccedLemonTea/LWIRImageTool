import scipy.constants as const
import scipy.integrate as integrate
import numpy as np
from .BlackbodyCalibration import BlackbodyCalibration

def NEdT_calculation(image_stack, calibration_coefficients, temperatures, wavelengths, response):

        plancks_constant = const.h  # 6.62607015e-34 [Joules*Seconds]
        speed_of_light_constant = const.c  # 299792458.0 [Meters/Second]
        boltzmann_constant = const.k  # 1.380649e-23 [Joules/Kelvin]

        wavelengths = wavelengths * 0.000001 # Microns --> Meters
        numerator = 2 * plancks_constant * speed_of_light_constant * speed_of_light_constant

        # Precompute wavelength-only constants (done ONCE)
        wl = wavelengths
        wl5 = wl**5

        hc_over_k = plancks_constant * speed_of_light_constant / boltzmann_constant

        radiance_image_stack = image_stack * calibration_coefficients[:,:,0,np.newaxis] + calibration_coefficients[:,:,1,np.newaxis]

        array_of_avg_coords = BlackbodyCalibration.find_ascensions(image_stack=image_stack, deriv_threshold=3, window_fraction=0.001)


        NEDT_array = np.zeros((image_stack.shape[0],image_stack.shape[1],len(temperatures)))
        for i, To in enumerate(temperatures):

            start = array_of_avg_coords[2 * i]
            stop = array_of_avg_coords[2 * i + 1]

            diffs = np.diff(radiance_image_stack[:,:,start:stop])

            sigma_diff = np.std(diffs, ddof=0)
            sigma = sigma_diff / np.sqrt(2)

            # Vectorized Planck derivative
            Xo = hc_over_k / (wl * To)
            expX = np.exp(Xo)

            numerator_dL = numerator * expX * Xo
            denominator = wl5 * (expX - 1)**2 * To

            dLdT = (numerator_dL / denominator) * response

            int_dLdT = integrate.simpson(dLdT, wavelengths)

            NEDT_array[:, :, i] = sigma / int_dLdT
        
        return NEDT_array