import cv2
import numpy as np
import matplotlib.pyplot as plt

from segmentation import BinaryThresholding, NormalizedOtsuThresholding

class Postprocessing():
    '''
    input to this method is a variable for the binary image
    '''
    def find_contours(bin_image):

        mask = np.zeros_like(bin_image)

        contour, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        longest_cntr = None
        largest_area = 0
        for cntr in contour:
            area = cv2.contourArea(cntr)    
            if area > largest_area:
                largest_area = area
                longest_cntr = cntr
        
        '''
        TODO: try to plot the longest contour to check if small melanoma will be ignoerd
        '''
        
        return longest_cntr
    
    
    def feature_extractrion(image_number, longest_cntr, image_shape, open_contour_threshold = 0.1):
        
        _,_,width,height = cv2.boundingRect(longest_cntr)

        #print("longest_cntr=", longest_cntr)

        first_tuple = (None, None, None, None)
        ghassan_tuple = (None)
        third_tuple = (None, None, None, None, None)
        
        if longest_cntr is not None:

            largest_area = cv2.contourArea(longest_cntr)
        
            rect_area = width*height

            ''' parametize the function ex. image_shape param np.zeros'''

            external_contours = np.zeros(image_shape) # as a black image
            

            ''' filtering open contours '''
            extent = float(largest_area)/rect_area

            if extent > open_contour_threshold:    

                mask = np.zeros_like(image_shape)

                cv2.drawContours(external_contours, [longest_cntr], -1, 255, 2)
                
                mask = cv2.drawContours(mask, [longest_cntr], -1, 255, -1)                        

                ''' M = Momentum '''
                M = cv2.moments(longest_cntr)

                if M["m00"] != 0:
                                                
                    #Center of Mass(com) of the contours                            
                    com_x = int(M['m10']/M['m00'])
                    com_y = int(M['m01']/M['m00'])
                                    
                    major_diameter = 2 * np.sqrt(M["mu20"] / M["m00"])
                    minor_diameter = 2 * np.sqrt(M["mu02"] / M["m00"])
                    
                '''since just closed contours are considered, the param @closed always set to True '''                
                perimeter = cv2.arcLength(longest_cntr, True)

                ''' Paper: Melanoma Skin Cancer Detection Using Image Processing and Machine Learning Techniques '''                                    
                ir_A = perimeter / largest_area                                     
                Circularity = perimeter**2 / largest_area                           
                circile_irrigularity = perimeter**2 / (4*largest_area*np.pi)        
                ir_Abnormality = (4*largest_area*np.pi)/perimeter                            
                            
                first_tuple = (ir_A, Circularity, circile_irrigularity, ir_Abnormality)

                # Fitting an Ellipse
                ellipse = cv2.fitEllipse(longest_cntr)

                ellipse_cnt = cv2.ellipse2Poly( (int(ellipse[0][0]),int(ellipse[0][1]) ) ,( int(ellipse[1][0]),int(ellipse[1][1]) ),int(ellipse[2]),0,360,1)

                ellipse_irrigularity = cv2.matchShapes(longest_cntr, ellipse_cnt, 1, 0.0)

                cv2.ellipse(external_contours,ellipse,(255),2)

                ghassan_tuple = (ellipse_irrigularity)

                ''' Paper: Computer aided Melanoma skin cancer detection using Image Processing ..'''            
                Circularity_indx = (4*largest_area*np.pi) / perimeter**2
                
                ir_A = perimeter / largest_area

                ir_B = perimeter / major_diameter
                
                ir_C = perimeter * (1/minor_diameter - 1/major_diameter)
                
                ir_D = major_diameter - minor_diameter            

                third_tuple = (ir_A, ir_B, ir_C, ir_D, Circularity_indx)
                
                # cv2.ellipse(external_contours,ellipse,255,2)
                cv2.circle(external_contours, (com_x, com_y), 5, 255, -1)
        
        else:
            #cv2.putText(external_contours,"Unable to detect closed contours!",(50,250),
            #            fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)

            print("Unable to detect closed contours to this image:!" + str(image_number))   
        

        return [first_tuple, ghassan_tuple, third_tuple]
