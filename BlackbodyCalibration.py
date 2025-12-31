### Blackbody Calibration Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 12/23/2025
# File : Blackbody.py


import LWIRimagetool

import os
import numpy as np

class BlackbodyCalibration(LWIRimagetool.CalibrationData):
    def __init__(self,directory,filetype,rsr):
        LWIRimagetool.CalibrationData.__init__(self)
        self.generate_coefficients(directory,filetype,rsr)


    def generate_coefficients(self,directory,filetype,rsr):
        """
        Calculates regional averages based on a blackbody run. These regional averages
        are then plotted against calculated blackbody radiances, linear regression is performed
        and the function provides gain and bias terms for each pixel in the image. 
        See details on how to perform a blackbody run in the README:
            directory(str): Directory containing images from blackbody run
        Returns:
        Array-Like: Array containing the calibration coefficients for each pixel in the detector

        """

        directory = os.fsencode(directory)
        first_image_path = None
        print(directory)
        # Loop through each image
        for file in sorted(os.listdir(directory)):
            filename = os.fsdecode(file)
            file_path = os.path.join(os.fsdecode(directory), filename)
            if first_image_path is None:
                Factory = LWIRimagetool.ImageDataFactory()
                first_src = Factory.create_from_file(file_path,filetype)
                image_stack = np.array(first_src.raw_counts)
                first_image_path = file_path
            else:
                src = Factory.create_from_file(file_path,filetype)
                image_stack = np.dstack((image_stack,src.raw_counts))
        
        for col in range(image_stack.shape(0)):
            for row in range(image_stack.shape(1)):
                
                ### SELECTING PIXEL AND GRABBING STATS ###
                # individual pixel being selected as a 1d vector, changing with time
                individual_pixel = src.raw_counts[row,col,:]

                # Averaging data to smooth 1st derivative
                chunk_size = int(individual_pixel.shape[0]*0.001)
                
                means = []
                for i in range(0,individual_pixel.shape[0], chunk_size):
                    chunk = individual_pixel[i:i+chunk_size]
                    means.append(chunk.mean())

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
                step_averages = np.array([])
                average_x_vals = np.array([])

                # Averaging between the end of one ascension and beginning of next
                # 2nd derivative min --> 2nd derivative max
                for i in range(0, array_of_avg_coords.shape[0]-1,2):
                    step_cum_sum = 0.0
                    count = 0.0
                    average_x_vals = np.append(average_x_vals,int((array_of_avg_coords[i] + array_of_avg_coords[i+1])/2)*chunk_size)

                    for j in range(array_of_avg_coords[i],array_of_avg_coords[i+1]-1,1):
                        if first_derivative[j] <= (3*stdev_first_deriv + mean_first_deriv):
                            for b in range(j*chunk_size,(j+1)*chunk_size,1):
                                step_cum_sum = step_cum_sum + individual_pixel[b]
                                count = count + 1
                    if count != 0:
                        step_cum_sum = float(step_cum_sum / count)   
                        step_averages = np.append(step_averages,[step_cum_sum])

                if rsr is not None:
                    ### GENERATING BLACKBODY BAND RADIANCES ###
                    blackbody = LWIRimagetool.Blackbody()
                    txt_content = np.loadtxt(file_path,skiprows=1,delimiter=',')
                    wavelengths = txt_content[:,0]
                    response = txt_content[:,1]
                    band_radiances = []
                    for i in range(step_averages.shape[0]):
                        temp = 283 + i*5.0 # Assumes that the blackbody run is moving by 5 degree steps -- may need to make this adjustable
                        blackbody.absolute_temperature = temp
                        band_radiances.append(blackbody.band_radiance(wavelengths,response))
                else:
                    ### GENERATING BLACKBODY BAND RADIANCES ###
                    blackbody = LWIRimagetool.Blackbody()
                    wavelengths = np.linspace(8, 14, 10000)
                    band_radiances = []
                    for i in range(step_averages.shape[0]):
                        temp = 283 + i*5.0 # Assumes that the blackbody run is moving by 5 degree steps -- may need to make this adjustable
                        blackbody.absolute_temperature = temp
                        band_radiances.append(blackbody.band_radiance(wavelengths))


                ### APPLYING LINEAR REGRESSION TO FIND GAIN AND BIAS TERMS ###
                gain, bias = np.polyfit(step_averages,band_radiances,1)
                cal_array = np.empty((src.raw_counts.shape[0],src.raw_counts.shape[1]))
                cal_array[row,col,0] = gain
                cal_array[row,col,1] = bias

        return cal_array
        
                
if __name__ == '__main__':
    import os
    import LWIRimagetool

    directory = "/home/cjw9009/Desktop/Senior_Project/FLIRSIRAS_CalData/20251110_1620"
    filetype = 'envi'
    
    Factory = LWIRimagetool.CalibrationDataFactory()
    Blackbody_Coefficients = Factory.create_from_file(directory, 'blackbody', 'rjpeg')
    

     