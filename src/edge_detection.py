'''
*Assignment 3 - Edge Detection*

Script for cropping the text in an image and higlighting letters' edge contours using blurring and canny edge detection.

creating visual environmet, executing code from the terminal.

The script only uses the regular requirements from the cv101 environment used for the cds-cv course.

If you don't have such an environment, you can run the line "bash create_vision_venv.sh"  from the terminal to create one.

Activate the environment by running "source cv101/bin/activate".

Go to the src folder of the repository by running cd src in the terminal.

Run the the python script by running "python edge_detection.py".

'''



# We need to include the home directory in our path, so we can read in our own module.
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np


def main(image_path = os.path.join("..", "data", "data_assignment3", "we_hold_these_truths.JPG")):
  
    image = cv2.imread(image_path) # read image from filepath
    
     #draw green rectangle around the text body. We have found the the dimension manually, but we would have liked to find a robust and generalizable method for this
    image_rectangular = cv2.rectangle(image.copy(), (1250, 850), (3000, 3000), (0,255,0), 10)
    
    cv2.imwrite(os.path.join("..","data","data_assignment3", "image_with_ROI.jpg"), image_rectangular) #save image
    print("The image image_with_ROI.jpg has been saved.")
    
    # crop the image
    image_cut = image.copy()[850:3000,1250:3000] #crop using same dimensions 
    cv2.imwrite(os.path.join("..","data","data_assignment3", "image_cropped.jpg"), image_cut)  #save image
    print("The image image_cropped.jpg has been saved.")
    
    # grey image
    grey_image = cv2.cvtColor(image_cut, cv2.COLOR_BGR2GRAY) #transform cropped image to greyscale
    
    # blur, canny...
    blurred_mini = cv2.medianBlur(grey_image,  5, 0) #blurr a bit usig median blurring
    
    #threshold image to remove tile lines. Threshold have been found manually by plotting the color intensity of the image
    (T, thres) = cv2.threshold(blurred_mini , 125, 255, cv2.THRESH_BINARY) 
    
    #blur even more to remove tile lines and to prepare for edge detection
    blurred = cv2.medianBlur(thres,  5, 0)
    
    #canny edge detection
    canny = cv2.Canny(blurred, 30, 150)
    
    #finding the contours around the letters
    (cnts, _) = cv2.findContours(canny, 
                             cv2.RETR_EXTERNAL, 
                             cv2.CHAIN_APPROX_SIMPLE)
    #draw the contours We choose a rather thick contour for this image
    image_contours = (cv2.drawContours(image_cut.copy(), cnts, -1, (0, 255, 0), 7))
    
    #Save image
    cv2.imwrite(os.path.join("..", "data","data_assignment3", "image_letters.jpg"), image_contours)  

    print("The image image_letters.jpg has been saved.")
    print("All images have succesfully been saved in data/data_assignment3/")

if __name__=="__main__":
    main()



