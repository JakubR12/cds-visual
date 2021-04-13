#!/usr/bin/env python

"""
Instructions to run a script

1. clone the repository 
2. upload the impressionist painting data manually into data/data_assignment5 in the cloned repo
3. unzip the file in the terminal: unzip filename.zip
2. in the terminal navigate into the cloned repo
3. create a virtual environment: bash create_cnn_venv.sh
4. activate the virtual environment: source cnn_venv/bin/activate
5. navigate to the src folder: cd src
6. run the script: python cnn-artists.py
7. the metric and model's history fit figure are saved into data/assignment5/out
8. Enjoy

"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
import os 


# sklearn tools
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Dropout, #layer to to kill some nodes at random. Sometimes helps with overfitting...
                                     BatchNormalization)  #layer to normalize weight-values. Sometimes helps with performance and overfitting...

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam # optimizer related to Stochastic gradient descent. Makes the model fit a little faster...
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping #makes the model stop training, when it stops improving
from tensorflow.keras.preprocessing.image import ImageDataGenerator #helps with loading data and generating more training data. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


#paths for data
training_dir = os.path.join("..", "data", "data_assignment5", "training", "training")
test_dir = os.path.join("..", "data", "data_assignment5", "validation", "validation")


def plot_history(H, epochs,file_name):
    """"
	Function for plotting keras model training history
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig(file_name)
  

def main():

	image_size = 64  #set the height and width of images
	# define model
	model = Sequential()

	# first set of CONV => BATCHNORM => RELU => POOL => DROP
	model.add(Conv2D(32, (3, 3), 
	                 padding="same", 
	                 input_shape=(image_size, image_size, 3)))
	model.add(BatchNormalization())#transforming layer output to the mean of 0 and SD 1 (a.k.a. good old scaling)
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), 
	                       strides=(2, 2)))
	#model.add(Dropout(0.2)) #prevents overfitting by randomly killing some nodes to prevent the netowrk from relying on a few overfitting nodes

	# second set of CONV => BATCHNORM => RELU => POOL => DROP
	model.add(Conv2D(64, (5, 5), 
	                 padding="same"))
	model.add(BatchNormalization()) 
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), 
	                       strides=(2, 2)))
	#model.add(Dropout(0.2))

	# FC => BATCHNORM => RELU => DROP
	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Dropout(0.2))


	# softmax classifier
	model.add(Dense(10))
	model.add(Activation("softmax"))



	# for a more detail explanation: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
	# formula for generating new train data 
	train_datagen = ImageDataGenerator(
	        rescale=1./255, #rescale pizel values to be between 0 and 1
	        shear_range=0.4, #randomly shift images to the side
	        zoom_range=0.4, #zoom a bit
	        horizontal_flip=True) #mirror images


	# rescaling the test data
	test_datagen = ImageDataGenerator(rescale=1./255)

	batch_size = 8

	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
	        training_dir,  # this is the target directory
	        target_size=(image_size, image_size),  # all images will be resized to 64 x 64
	        batch_size=batch_size,
	        class_mode='categorical', #since we use categorical_crossentropy loss, we need binary labels
	        interpolation = "lanczos")  #method for resizing picture that supposed to yield the highest accuracy

	# this is a similar generator, for test data (no new images added)
	validation_generator = test_datagen.flow_from_directory(
	        test_dir,
	        target_size=(image_size, image_size),
	        batch_size=batch_size,
	        class_mode='categorical',
	        interpolation = "lanczos")

	# argument for early stopping of the model once it is not improving anymore
	callback = EarlyStopping(monitor="loss", patience=3, min_delta = 0.001)
	
	# the optimizer Adam is an optimization of the stochastic gradient descent (better results in shorter time)
	opt = Adam()
	model.compile(loss="categorical_crossentropy",
	              optimizer= opt,
	              metrics=["accuracy"])


	batch_size = 32
	epochs = 10
	# fit model and save fitting hisotry
	H = model.fit(
	        train_generator, #use the image generator
	        epochs=epochs,
	        batch_size = batch_size, 
	        steps_per_epoch = 4000//batch_size, #set step size, so we run throught the full datasize each epoch only once
	        validation_data=validation_generator, #use the resized test data
	        validation_steps = 1000//batch_size,  
	              verbose=1,
	              callbacks=[callback]
	        )

	model.save_weights(os.path.join("..","data", "data_assignment5", "weights","model.h5")) #save the weights for another time

	plot_history(H,epochs,os.path.join("..","data", "data_assignment5", "out","history_plot.png")) #plot the training hisory


	# get the ground truth of your data. 
	test_labels=validation_generator.classes 

	# predict the probability distribution of the data
	predictions=model.predict(validation_generator, verbose=1)

	# get the class with highest probability for each sample
	y_pred = np.argmax(predictions, axis=-1)

	# get the classification report
	cr = classification_report(test_labels, y_pred)
	print(cr)
	filepath = os.path.join("..","data","data_assignment5","out","metrics.txt")
	text_file = open(filepath, "w")
	text_file.write(cr)
	text_file.close()

if __name__=="__main__":
	main()
