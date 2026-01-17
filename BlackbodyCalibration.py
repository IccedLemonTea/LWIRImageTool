### Blackbody Calibration Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 12/23/2025
# File : Blackbody.py



import numpy as np
from .BlackbodyCalibrationConfig import BlackbodyCalibrationConfig
from .CalibrationData import CalibrationData
from .StackImages import stack_images
from .Blackbody import Blackbody


class BlackbodyCalibration(CalibrationData):
    """
    Performs blackbody calibration of LWIR images.

    This class reads in a sequence of blackbody images, detects temperature 
    ascensions, and calculates gain and bias coefficients for each pixel.
    """

    def __init__(self, config : BlackbodyCalibrationConfig):
        """
        Initializes the calibration by stacking images, detecting ascensions,
        and generating calibration coefficients.
        """
        CalibrationData.__init__(self)
        self.image_stack = stack_images(config.directory, config.filetype, config.progress_cb)
        _array_of_avg_coords = self.find_ascensions(self.image_stack, config.deriv_threshold, config.window_fraction, config.progress_cb)
        self.coefficients = self.generate_coefficients(self.image_stack, _array_of_avg_coords, config.blackbody_temperature, config.temperature_step, config.rsr, config.progress_cb)

    def find_ascensions(self, image_stack, deriv_threshold, window_fraction, progress_cb):
        """
        Detects frame indices corresponding to temperature steps (ascensions) 
        using the mean signal over all pixels.

        Parameters
        ----------
        image_stack : np.ndarray
            3D array of stacked images (rows, cols, num_frames).
        chunk_percentage : float
            Fraction of signal length used for chunk averaging to smooth data.
        deriv_threshold : float
            Multiplier for standard deviation when detecting derivative peaks.
        window_fraction : float
            Fraction of data length for matching derivative peaks to temperature changes.

        Returns
        -------
        array_of_avg_coords : np.ndarray
            Array of start and end indices of temperature steps in frames.
        """

        # Optional: update GUI progress after each step
        if progress_cb:
            progress_cb(phase="ascension", current=0, total=1)

        ### CALCULATING STATISTICS ###
        means = np.mean(image_stack,axis=(0,1))
        first_derivative = np.gradient(means)
        second_derivative = np.gradient(first_derivative)

        stdev_first_deriv = np.std(first_derivative)
        stdev_second_deriv = np.std(second_derivative)
        mean_first_deriv = np.mean(first_derivative)
        mean_second_deriv = np.mean(second_derivative)

        ### CALCULATING THE REGIONS OF ASCENSION ###
        # Vector to hold all values of when the 1st derivative exceeds 3 stdevs of the mean
        # Means that the DC of the scene is changing --> new temperature being reached in the cal run 
        # e.g. ascends to a new temperature
        change_in_temp = [0]

        for i in range(first_derivative.shape[0]):
            if first_derivative[i] >= (deriv_threshold*stdev_first_deriv + mean_first_deriv):
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
            if second_derivative[i] >= (deriv_threshold*stdev_second_deriv + mean_second_deriv):
                ascension_start.append(i)
            if second_derivative[i] <= (-deriv_threshold*stdev_second_deriv + mean_second_deriv):
                ascension_end.append(i)

        # Window searching 1% of the data size
        window = int(len(means))*window_fraction 
        ascensions = False
        # print(f"ascenscion start size{len(ascension_start)} ascenscion end size{len(ascension_end)}")
        # Finding the max and min frame counts of the ascension
        for i in range(len(change_in_temp)-1):
            temp_ascension = []
            for j in range(len(ascension_start)-1):
                if (change_in_temp[i] + window) >= ascension_start[j] and (change_in_temp[i] - window <= ascension_start[j]):
                    temp_ascension.append(ascension_start[j])
            for j in range(len(ascension_end)-1):
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
        print(f"ascensions calculated")
        return array_of_avg_coords

    def generate_coefficients(self, image_stack, array_of_avg_coords, blackbody_temperature, tempurature_step, rsr, progress_cb):
        """
        Calculates gain and bias coefficients for each pixel in a fully vectorized manner.

        Regional averages are computed between ascension intervals, and linear regression 
        is applied to map counts to radiance.

        Parameters
        ----------
        image_stack : np.ndarray
            3D array of stacked images (rows, cols, num_frames).
        array_of_avg_coords : np.ndarray
            Array of start and end indices of temperature steps (ascensions).
        rsr : str or None
            Path to RSR file for spectral weighting. If None, default wavelengths are used.
        progress_cb : callable, optional
            Callback function for GUI progress updates.

        Returns
        -------
        cal_array : np.ndarray
            3D array of calibration coefficients with shape (rows, cols, 2)
            where [:,:,0] is gain and [:,:,1] is bias.
        """

        # Optional: update GUI progress after each step
        if progress_cb:
            progress_cb(phase="computing_steps", current=0, total=1)

        ### COMPUTING STEP AVERAGES FOR EACH PIXEL ###
        rows, cols, frames = image_stack.shape
        n_steps = len(array_of_avg_coords) // 2

        # Preallocate array size
        step_averages = np.zeros((rows, cols, n_steps))
        cal_array = np.zeros((rows,cols,2))

        for step in range(n_steps):
            start = array_of_avg_coords[2*step]
            end = array_of_avg_coords[2*step + 1]

            # Compute mean over time for all pixels in the step window
            step_averages[:, :, step] = np.mean(image_stack[:, :, start:end],axis=2)

            # Optional: update GUI progress after each step
            if progress_cb:
                progress_cb(phase="computing_steps", current=step+1, total=n_steps)
            print(f"computing step {(step+1)/n_steps}")

        # Determining usage of Relative Spectral Response Function
        if rsr:
            txt_content = np.loadtxt(rsr, skiprows=1, delimiter=',')
            wavelengths = txt_content[:, 0]
            response = txt_content[:, 1]
        else:
            wavelengths = np.linspace(8, 14, 10000)

        ### GENERATING BAND RADIANCES FOR EACH TEMP STEP ###
        band_radiances = np.zeros(n_steps)
        temperatures = blackbody_temperature + np.arange(n_steps) * tempurature_step # [K]
        bb = Blackbody()

        for i, temp in enumerate(temperatures):
            bb.absolute_temperature = temp # [K]
            if rsr:
                band_radiances[i] = bb.band_radiance(wavelengths, response)
            else:
                band_radiances[i] = bb.band_radiance(wavelengths)

        ### PERFORMING LINEAR REGRESSION ###
        for row in range(image_stack.shape[0]):
            for col in range(image_stack.shape[1]):
                gain, bias = np.polyfit(step_averages[row,col,:], band_radiances[:], 1)
                cal_array[row, col, 0] = gain
                cal_array[row, col, 1] = bias

        return cal_array
    
if __name__ == "__main__":
    import numpy as np
    from .BlackbodyCalibrationConfig import BlackbodyCalibrationConfig
    from .CalibrationDataFactory import CalibrationDataFactory

    ### USER TEST CONFIG ###
    test_directory = "/home/cjw9009/Desktop/Senior_Project/FLIRSIRAS_CalData/20251202_1400"
    test_filetype = "rjpeg"
    test_rsr = "/home/cjw9009/Desktop/Senior_Project/FLIRSIRAS_CalData/flir_boson_with_13mm_45fov.txt"

    print("Starting BlackbodyCalibration test...")

    # Build validated config 
    config = BlackbodyCalibrationConfig(
        directory=test_directory,
        filetype=test_filetype,
        blackbody_temperature=283.15,   # K
        temperature_step=5.0,           # K
        rsr=test_rsr,
        progress_cb=None
    )
    print("Config Validated")
    # Create calibration via factory
    calib = CalibrationDataFactory.create(config)

    print("Calibration object created successfully.")
    print(f"Image stack shape: {calib.image_stack.shape}")
    print(f"Coefficient array shape: {calib.coefficients.shape}")

    # Save coefficients
    np.save(
        "2025.npy",
        calib.coefficients
    )

    print("Calibration coefficients saved.")

