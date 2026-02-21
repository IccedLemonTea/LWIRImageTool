
import numpy as np
from .BlackbodyCalibrationConfig import BlackbodyCalibrationConfig
from .CalibrationData import CalibrationData
from .StackImages import stack_images
from .Blackbody import Blackbody


class BlackbodyCalibration(CalibrationData):
    """
    Blackbody calibration of LWIR imagery.

    Stacks a sequence of blackbody images, detects temperature-step
    boundaries (ascensions), and fits a per-pixel linear model mapping
    digital counts to band-integrated radiance.

    The calibrated coefficients satisfy::

        radiance[r, c] = gain[r, c] * DC[r, c] + bias[r, c]

    where ``gain = coefficients[r, c, 0]`` and
    ``bias = coefficients[r, c, 1]``.

    Parameters
    ----------
    config : BlackbodyCalibrationConfig
        Fully validated configuration object.

    Attributes
    ----------
    image_stack : np.ndarray
        Raw stacked images, shape ``(rows, cols, frames)``.
    coefficients : np.ndarray
        Per-pixel gain/bias array, shape ``(rows, cols, 2)``.
    directory : str
        Source directory used for this calibration run.
    blackbody_temperature : float
        Starting blackbody temperature [K].
    temperature_step : float
        Temperature increment per step [K].
    rsr : str, np.ndarray, or None
        RSR definition used (mirrors ``config.rsr``).
    deriv_threshold : float
        Derivative threshold multiplier used for ascension detection.
    window_fraction : float
        Window fraction used for ascension detection.

    See Also
    --------
    BlackbodyCalibrationConfig : Configuration dataclass for this class.
    CalibrationDataFactory : Recommended entry point for construction.
    """

    def __init__(self, config: BlackbodyCalibrationConfig):
        """
        Initializes the calibration by stacking images, detecting ascensions,
        and generating calibration coefficients.
        """
        CalibrationData.__init__(self)
        self.image_stack = stack_images(config.directory, config.filetype,
                                        config.progress_cb)
        _array_of_avg_coords = self.find_ascensions(self.image_stack,
                                                    config.deriv_threshold,
                                                    config.window_fraction,
                                                    config.progress_cb)
        self.coefficients = self.generate_coefficients(
            self.image_stack, _array_of_avg_coords,
            config.blackbody_temperature, config.temperature_step, config.rsr,
            config.progress_cb)
        
        # Asigning config vars to object os users can see how cal was created.
        self.directory             = config.directory
        self.blackbody_temperature = config.blackbody_temperature
        self.temperature_step      = config.temperature_step
        self.rsr                   = config.rsr
        self.deriv_threshold       = config.deriv_threshold
        self.window_fraction       = config.window_fraction

    def find_ascensions(self,
                        image_stack,
                        deriv_threshold=3,
                        window_fraction=0.001,
                        progress_cb=None):
        """
        Detect frame indices bounding each temperature step (ascension).

        Computes the spatial mean signal over all pixels for every frame,
        then uses first- and second-derivative thresholding to locate the
        start and end of each temperature transition.

        Parameters
        ----------
        image_stack : np.ndarray
            Stacked images, shape ``(rows, cols, frames)``.
        deriv_threshold : float, optional
            Standard-deviation multiplier for first-derivative peak
            detection.  Default is ``3``.
        window_fraction : float, optional
            Fraction of total frames used as the search window when
            associating derivative peaks with temperature steps.
            Default is ``0.001``.
        progress_cb : callable or None, optional
            Progress callback.  Default is ``None``.

        Returns
        -------
        array_of_avg_coords : np.ndarray
            1-D array of frame indices alternating between step-start and
            step-end positions.  Length is ``2 * n_steps + 1`` (the final
            element is the last frame index).
        """

        # Optional: update GUI progress after each step
        if progress_cb:
            progress_cb(phase="ascension", current=0, total=1)

        ### CALCULATING STATISTICS ###
        means = np.mean(image_stack, axis=(0, 1))
        first_derivative = np.gradient(means)
        second_derivative = np.gradient(first_derivative)

        stdev_first_deriv = np.std(first_derivative)
        stdev_second_deriv = np.std(second_derivative)
        mean_first_deriv = np.mean(first_derivative)
        mean_second_deriv = np.mean(second_derivative)

        # print(f"The mean of the first deriv is {mean_first_deriv}, and {mean_second_deriv} for the second deriv")
        # print(f"99% Of the data lies within {3*stdev_first_deriv} for the first deriv, and {3*stdev_second_deriv} for the second deriv")


        ### CALCULATING THE REGIONS OF ASCENSION ###
        # Vector to hold all values of when the 1st derivative exceeds 3 stdevs of the mean
        # Means that the DC of the scene is changing --> new temperature being reached in the cal run 
        # e.g. ascends to a new temperature
        change_in_temp = [0]

        for i in range(first_derivative.shape[0]):
            if first_derivative[i] >= (3*stdev_first_deriv + mean_first_deriv):
                if change_in_temp is not None:
                    change_in_temp.append(i)
                else:
                    change_in_temp = []
                    change_in_temp.append(i)

        # Adding end point of derivative vector
        change_in_temp.append(first_derivative.shape[0])


        # Vector to hold all derivative values that 
        # signal the beginning and end of the temperature change 
        # portion of the blackbody run (ASCENSION)
        ascension_start = []
        ascension_end = []
        for i in range(second_derivative.shape[0]):
            if second_derivative[i] >= (3*stdev_second_deriv + mean_second_deriv):
                ascension_start.append(i)
            if second_derivative[i] <= (-3*stdev_second_deriv + mean_second_deriv):
                ascension_end.append(i)

        # Window searching 1% of the data size
        window = int(len(means))*0.01 
        ascensions = False
        # print(f"ascenscion start size{len(ascension_start)} ascenscion end size{len(ascension_end)}")
        # Finding the max and min frame counts of the ascension
        for i in range(len(change_in_temp)-1):
            temp_ascension = []
            for j in range(len(ascension_start)):
                if (change_in_temp[i] + window) >= ascension_start[j] and (change_in_temp[i] - window <= ascension_start[j]):
                    temp_ascension.append(ascension_start[j])
            for j in range(len(ascension_end)):
                if (change_in_temp[i] + window) >= ascension_end[j] and (change_in_temp[i] - window <= ascension_end[j]):
                    temp_ascension.append(ascension_end[j])

            if temp_ascension == []:
                continue
            else:
                begin_average = min(temp_ascension)
                end_average = max(temp_ascension)
                if (change_in_temp[i+1] > change_in_temp[i] + window):
                    if ascensions == False:
                        array_of_avg_coords = np.array([0, begin_average, end_average])
                        ascensions = True
                    else:
                        array_of_avg_coords = np.append(array_of_avg_coords,[begin_average,end_average])

        array_of_avg_coords = np.append(array_of_avg_coords, len(means))
        if progress_cb:
            progress_cb(phase="ascension", current=1, total=1)
        return array_of_avg_coords

    def generate_coefficients(self, image_stack, array_of_avg_coords,
                              blackbody_temperature, tempurature_step, rsr,
                              progress_cb):
        """
        Fit per-pixel gain and bias coefficients via linear regression.

        For each pixel, computes the mean digital count within each
        temperature-step window, integrates the Planck function against the
        RSR to get band radiance at each step temperature, then fits a
        first-order polynomial (least squares) mapping counts to radiance.

        Parameters
        ----------
        image_stack : np.ndarray
            Stacked images, shape ``(rows, cols, frames)``.
        array_of_avg_coords : np.ndarray
            Step boundary indices from ``find_ascensions()``.
        blackbody_temperature : float
            Starting temperature [K].
        temperature_step : float
            Temperature increment per step [K].
        rsr : str, np.ndarray, or None
            RSR definition (see ``BlackbodyCalibrationConfig.rsr``).
        progress_cb : callable or None
            Progress callback.

        Returns
        -------
        cal_array : np.ndarray
            Per-pixel coefficients, shape ``(rows, cols, 2)``.
            ``cal_array[:, :, 0]`` is gain, ``cal_array[:, :, 1]`` is bias.
        """

        # Optional: update GUI progress after each step
        if progress_cb:
            progress_cb(phase="computing_steps", current=0, total=1)

        ### COMPUTING STEP AVERAGES FOR EACH PIXEL ###
        rows, cols, frames = image_stack.shape
        n_steps = len(array_of_avg_coords) // 2

        # Preallocate array size
        step_averages = np.zeros((rows, cols, n_steps))
        cal_array = np.zeros((rows, cols, 2))

        for step in range(n_steps):
            start = array_of_avg_coords[2 * step]
            end = array_of_avg_coords[2 * step + 1]

            # Compute mean over time for all pixels in the step window
            step_averages[:, :, step] = np.mean(image_stack[:, :, start:end],
                                                axis=2)

            # Optional: update GUI progress after each step
            if progress_cb:
                progress_cb(phase="computing_steps",
                            current=step + 1,
                            total=n_steps)

        if isinstance(rsr, str):
            txt_content = np.loadtxt(rsr, skiprows=1, delimiter=',')
            wavelengths = txt_content[:, 0]
            response = txt_content[:, 1]
        elif isinstance(rsr, np.ndarray):
            wavelengths = rsr[0, :]
            response = rsr[1, :]
        else:
            wavelengths = np.linspace(8, 14, 10000)
            response = np.ones_like(wavelengths)


        ### GENERATING BAND RADIANCES FOR EACH TEMP STEP ###
        band_radiances = np.zeros(n_steps)
        temperatures = blackbody_temperature + np.arange(
            n_steps) * tempurature_step  # [K]
        bb = Blackbody()

        for i, temp in enumerate(temperatures):
            bb.absolute_temperature = temp  # [K]
            band_radiances[i] = bb.band_radiance(wavelengths, response)

        ### PERFORMING LINEAR REGRESSION ###
        for row in range(image_stack.shape[0]):
            for col in range(image_stack.shape[1]):
                gain, bias = np.polyfit(step_averages[row, col, :],
                                        band_radiances[:], 1)
                cal_array[row, col, 0] = gain
                cal_array[row, col, 1] = bias

        return cal_array


if __name__ == "__main__":
    import numpy as np
    from .BlackbodyCalibrationConfig import BlackbodyCalibrationConfig
    from .CalibrationDataFactory import CalibrationDataFactory
    from .StackImages import stack_images
    import scipy.integrate as integrate
    import matplotlib.pyplot as plt
    import scipy.constants as const
    import math

    txt_content = np.loadtxt(
        "/home/cjw9009/Desktop/suas_data/flir_boson_with_13mm_45fov.txt",
        skiprows=1,
        delimiter=',')
    wavelengths = txt_content[:, 0]
    response = txt_content[:, 1]

    ### USER TEST CONFIG ###
    test_directory = "/home/cjw9009/Desktop/suas_data/FLIRSIRAS_CalData/20251202_1400"
    test_filetype = "rjpeg"
    test_rsr = "/home/cjw9009/Desktop/suas_data/flir_boson_with_13mm_45fov.txt"

    print("Starting BlackbodyCalibration test...")

    # Build validated config
    config = BlackbodyCalibrationConfig(
        directory=test_directory,
        filetype=test_filetype,
        blackbody_temperature=283.15,  # K
        temperature_step=5.0,  # K
        rsr=test_rsr,
        progress_cb=None)
    print("Config Validated")
    # Create calibration via factory
    calib = CalibrationDataFactory.create(config)

    print("Calibration object created successfully.")
    print(f"Image stack shape: {calib.image_stack.shape}")
    print(f"Coefficient array shape: {calib.coefficients.shape}")

    # Save coefficients
    np.save("2025.npy", calib.coefficients)

    stack = calib.image_stack
    array_of_avg_coords = calib.find_ascensions(stack, 3, 0.001, [])
    # multiply DC by gain, add bias to get per pixel radiance

    # NEDT Calculation ### ADD CODE HERE

    temps = [
        283.15, 288.15, 293.15, 298.15, 303.15, 308.15, 313.15, 318.15, 323.15,
        328.15, 333.15, 338.15, 343.15
    ]
    NEDT_array = np.empty((stack.shape[0], stack.shape[1], len(temps)))
    plancks_constant = const.h  # 6.62607015e-34 [Joules*Seconds]
    speed_of_light_constant = const.c  # 299792458.0 [Meters/Second]
    boltzmann_constant = const.k  # 1.380649e-23 [Joules/Kelvin]

    wavelengths = wavelengths * 0.000001
    numerator = 2 * plancks_constant * speed_of_light_constant * speed_of_light_constant

    # Precompute wavelength-only constants (done ONCE)
    wl = wavelengths
    wl5 = wl**5
    resp = response

    hc_over_k = plancks_constant * speed_of_light_constant / boltzmann_constant

    for r in range(stack.shape[0]):
        for c in range(stack.shape[1]):

            individual_pixel = stack[r, c, :]
            individual_pixel_rad = individual_pixel * calib.coefficients[
                r, c, 0] + calib.coefficients[r, c, 1]
            pixel_rad = individual_pixel_rad  # alias (faster lookup)

            for i, To in enumerate(temps):

                start = array_of_avg_coords[2 * i]
                stop = array_of_avg_coords[2 * i + 1]

                diffs = np.diff(pixel_rad[start:stop])

                avg_diff = np.mean(diffs)
                sigma_diff = np.std(diffs, ddof=0)
                sigma = sigma_diff / np.sqrt(2)

                # Vectorized Planck derivative
                Xo = hc_over_k / (wl * To)
                expX = np.exp(Xo)

                numerator_dL = numerator * expX * Xo
                denominator = wl5 * (expX - 1)**2 * To

                dLdT = (numerator_dL / denominator) * resp

                int_dLdT = integrate.simpson(dLdT, wavelengths)

                NEDT_array[r, c, i] = sigma / int_dLdT

    print(f"{NEDT_array.shape}")

    np.save("20251202_1400_fullimage_bbrun_NEDT_array_chat.npy", NEDT_array)
    mean_NEDT = np.mean(NEDT_array, axis=(0, 1))
    print(f"Size of mean_NEDT{mean_NEDT.shape}")
    plt.scatter(range(10, 71, 5), mean_NEDT)
    plt.title("Average NEDT at each step")
    plt.xlabel("Temperature in Kelvin (Step Temperature)")
    plt.ylabel("Mean NEDT")
    plt.savefig("Plot of updated NEDT")
    plt.show()