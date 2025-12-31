### Blackbody Calibration Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 12/23/2025
# File : Blackbody.py


import LWIRimagetool
import os
import numpy as np

class BlackbodyCalibration(LWIRimagetool.CalibrationData):
    """
    Performs blackbody calibration of LWIR images. 

    This class reads in a sequence of blackbody images, detects temperature 
    ascensions, and calculates gain and bias coefficients for each pixel.
    """

    def __init__(self, directory, filetype, rsr, progress_cb=None):
        """
        Initializes the calibration by stacking images, detecting ascensions,
        and generating calibration coefficients.

        Parameters
        ----------
        directory : str
            Path to the directory containing blackbody images.
        filetype : str
            Type/format of the image files (e.g., 'rjpeg', 'envi').
        rsr : str or None
            Path to the RSR file containing spectral response and wavelengths. If None, default wavelengths are used.
        progress_cb : callable, optional
            Callback function for progress updates. Called with `phase`, `current`, and `total`.
        """
        LWIRimagetool.CalibrationData.__init__(self)
        _image_stack = self.stack_images(directory, filetype, progress_cb)
        _array_of_avg_coords = self.find_ascensions(_image_stack, 0.01, 3, 0.001)
        self.generate_coefficients(_image_stack, _array_of_avg_coords, rsr, progress_cb)

    def stack_images(self, directory, filetype, progress_cb=None):
        """
        Reads all images from a directory and stacks them along a third dimension.

        The third dimension corresponds to time, ordered by the file timestamps.

        Parameters
        ----------
        directory : str
            Path to the directory containing blackbody images.
        filetype : str
            Type/format of the image files.
        progress_cb : callable, optional
            Callback function for GUI progress updates.

        Returns
        -------
        image_stack : np.ndarray
            3D array containing stacked images with shape (rows, cols, num_frames).
        """
        directory = os.fsencode(directory)
        first_image_path = None
        total_files = len(os.listdir(directory))
        idx = 0
        for file in sorted(os.listdir(directory)):
            filename = os.fsdecode(file)
            file_path = os.path.join(os.fsdecode(directory), filename)
            if first_image_path is None:
                Factory = LWIRimagetool.ImageDataFactory()
                first_src = Factory.create_from_file(file_path, filetype)
                image_stack = np.array(first_src.raw_counts)
                first_image_path = file_path
                if progress_cb:
                    progress_cb(phase="loading", current=idx + 1, total=total_files)
                    idx += 1
            else:
                src = Factory.create_from_file(file_path, filetype)
                image_stack = np.dstack((image_stack, src.raw_counts))
                if progress_cb:
                    progress_cb(phase="loading", current=idx + 1, total=total_files)
                    idx += 1
        return image_stack

    def find_ascensions(self, image_stack, chunk_percentage, deriv_threshold, window_fraction):
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
        means_of_stack = np.mean(image_stack, axis=(0, 1))
        chunk_size = int(means_of_stack.shape[0] * chunk_percentage)
        means = [means_of_stack[i:i + chunk_size].mean() for i in range(0, means_of_stack.shape[0], chunk_size)]

        first_derivative = np.gradient(means)
        second_derivative = np.gradient(first_derivative)

        stdev_first_deriv = np.std(first_derivative)
        stdev_second_deriv = np.std(second_derivative)
        mean_first_deriv = np.mean(first_derivative)
        mean_second_deriv = np.mean(second_derivative)

        change_in_temp = [0]
        for i, val in enumerate(first_derivative):
            if val >= (deriv_threshold * stdev_first_deriv + mean_first_deriv):
                change_in_temp.append(i)
        change_in_temp.append(len(first_derivative))

        ascension_start = [i for i, val in enumerate(second_derivative) if val >= (deriv_threshold * stdev_second_deriv + mean_second_deriv)]
        ascension_end = [i for i, val in enumerate(second_derivative) if val <= (-deriv_threshold * stdev_second_deriv + mean_second_deriv)]

        window = int(len(means) * window_fraction)
        ascensions = False
        array_of_avg_coords = []

        for i in range(len(change_in_temp) - 1):
            temp_ascension = []
            for start in ascension_start:
                if abs(change_in_temp[i] - start) <= window:
                    temp_ascension.append(start)
            for end in ascension_end:
                if abs(change_in_temp[i] - end) <= window:
                    temp_ascension.append(end)
            if temp_ascension:
                begin_average = min(temp_ascension)
                end_average = max(temp_ascension)
                if not ascensions:
                    array_of_avg_coords = np.array([0, begin_average, end_average])
                    ascensions = True
                else:
                    array_of_avg_coords = np.append(array_of_avg_coords, [begin_average, end_average])
        array_of_avg_coords = np.append(array_of_avg_coords, len(means))
        return array_of_avg_coords

    def generate_coefficients(self, image_stack, array_of_avg_coords, rsr, progress_cb=None):
        """
        Calculates gain and bias coefficients for each pixel based on blackbody ascensions.

        Regional averages are computed between ascension intervals, and linear regression 
        is applied to map counts to radiance.

        Parameters
        ----------
        image_stack : np.ndarray
            3D array of stacked images (rows, cols, num_frames).
        array_of_avg_coords : np.ndarray
            Array of start and end indices of temperature steps.
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
        total_pixels = image_stack.shape[0] * image_stack.shape[1]
        pixel_count = 0
        cal_array = np.empty((image_stack.shape[0], image_stack.shape[1], 2))
        for col in range(image_stack.shape[0]):
            for row in range(image_stack.shape[1]):
                pixel_count += 1
                individual_pixel = image_stack[row, col, :]
                first_derivative = np.gradient(individual_pixel)
                stdev_first_deriv = np.std(first_derivative)
                mean_first_deriv = np.mean(first_derivative)
                chunk_size = int(individual_pixel.shape[0] * 0.001)

                step_averages = np.array([])
                for i in range(0, array_of_avg_coords.shape[0] - 1, 2):
                    step_cum_sum = 0.0
                    count = 0.0
                    for j in range(array_of_avg_coords[i], array_of_avg_coords[i + 1] - 1):
                        if first_derivative[j] <= (3 * stdev_first_deriv + mean_first_deriv):
                            for b in range(j * chunk_size, (j + 1) * chunk_size):
                                step_cum_sum += individual_pixel[b]
                                count += 1
                    if count != 0:
                        step_cum_sum /= count
                        step_averages = np.append(step_averages, [step_cum_sum])

                # Generate band radiances
                blackbody = LWIRimagetool.Blackbody()
                if rsr is not None:
                    txt_content = np.loadtxt(rsr, skiprows=1, delimiter=',')
                    wavelengths = txt_content[:, 0]
                    response = txt_content[:, 1]
                    band_radiances = [blackbody.band_radiance(wavelengths, response) for i in range(len(step_averages))]
                else:
                    wavelengths = np.linspace(8, 14, 10000)
                    band_radiances = [blackbody.band_radiance(wavelengths) for i in range(len(step_averages))]

                gain, bias = np.polyfit(step_averages, band_radiances, 1)
                cal_array[row, col, 0] = gain
                cal_array[row, col, 1] = bias

                if progress_cb and pixel_count % 500 == 0:
                    progress_cb(phase="calibrating", current=pixel_count, total=total_pixels)

        return cal_array
