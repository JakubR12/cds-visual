Assignment 3 - Edge Detection 
==============================
#### Peter Thramkrongart and Jakub Raszka

### Reproducibility

The results are reproducible by cloning the repository (link above) and running the edge_detection.py script in the virtual environment called create_edge_detection_venv. Required packages can be found in the requirements.txt file. Alternatively, while being in the assignment-3 folder, you can just run the edge_detection.sh script which will automatically create a virtual environment, run the script and kill the environment to save your time. The pipeline was tested only on a linux server.

###  Description

The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation. The instructions are:
*	Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as __image_with_ROI.jpg__.
*	Crop the original image to create a new image containing only the ROI in the rectangle. Save this as __image_cropped.jpg__.
*	Using this cropped image, use Canny edge detection to 'find' every letter in the image
*	Draw a green contour around each letter in the cropped image. Save this as __image_letters.jpg__


### Results

We managed to highlight text in a given picture using computer vision. However, as can be seen, the result is far from perfect. Some of the brick interstices were picked up along with the text. Also, the approach relies on manual cropping and the blur parameters are hardly generalizable outside this assignment. Using object-character recognition, such as Tesseract along with a smart pre-processing and natural language processing software to predict the most plausible outcome would yield much better accuracy. 





<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
