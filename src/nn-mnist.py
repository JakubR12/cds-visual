#!/usr/bin/python

# importing modules
import argparse
import os
import sys
sys.path.append(os.path.join(".."))
import numpy as np

# Import utils for classification, preprocessing
from utils.neuralnetwork import NeuralNetwork
import utils.ml_preprocessing as mlp  # Custom module we made to load and split the mnist data

# Import sklearn metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


'''
# Instructions
To run the script follow these steps:
1. clone the repository: git clone https://github.com/JakubR12/cds-visual.git
2. navigate to the newly created repository
3. if you haven't upadted your cv101 environment, you can do so by: bash create_vision_venv.sh
4. activate the virtual environment: source cv101/bin/activate
5. go to the src folder: cd src
6  run the script: python lr-mnist.py

There are 7 arguments which can but do not have to be specified:

flags: -tr, --train_size,  default: 0.8:  description: int or float, a proportion of the data the model to be trained on
flags: -ts, --test_size:  default: 0.2, description: int or float, a proportion of the data the model to be trained on
flags: -r, --random_state: default: None, description:   int, a random seed
flags: -sm, --save_metrics: default: None, description: bool, whether to save the metrics
flags: -mf, --metrics_safe, default: neural_network_metrics.txt, description: str, the filename of the metrics with the .txt ending
flags: -e, --epochs, default: 80, description: int, a number of epochs
flgas: -hl, --hidden_layers, default: [32,16], description: list, a number of nodes in the two hidden layers, the third is set to 10 automatically

examples:
  python nn-mnist.py -tr 0.7 -ts 0.3 -r 2 -sm -mf neural_network_filename.txt -e 100 -hl 28, 14
  
  When using boolean flags (-sm), just leave it empty.

'''

def main(train_size = 0.8,
         test_size = 0.2, 
         random_state = None, 
         save_metrics = True, 
         metrics_filename = "neural_network_metrics.txt",
         hidden_layers = [32,16],
        epochs = 80):
    
    print("Preparing data. This may take a while ...")
    
    # using a function we built in the ml_preprocessing.py script to load, format, split and scale the data -> we get a clean data
    X_train, X_test, y_train, y_test = mlp.fetch_visual_data(train_size = train_size, test_size = test_size, random_state = random_state)
    
    print("The MNIST dataset has been loaded, split and scaled.")
    
    # convert labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)

    # train network
    print("[INFO] training network (~ 7 mins) ...")
    
    # specifying 
    hidden_layers.insert(0,X_train.shape[1])
    hidden_layers.append(10)
    print( hidden_layers)
    
    # input layer = data size, n of nodes in first layer, ..., n of output
    nn = NeuralNetwork( hidden_layers)
    
    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=epochs)

    # evaluate network
    print(["[INFO] evaluating network (around 7 minutes) ..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    cr = classification_report(y_test.argmax(axis=1), predictions)
    print(cr)


    
    # optional argument to save the data in the out file as a text file
    if save_metrics == True: 
        filepath = os.path.join("..","data","data_assignment4","out",metrics_filename)
        text_file = open(filepath, "w")
        text_file.write(cr)
        text_file.close()

    
if __name__=="__main__":
    
    # define comman line interface arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-tr", "--train_size", required = False, type = float, default = 0.5, 
                    help = "int or float: A proportion of the data the model to be trained on")
    ap.add_argument("-ts", "--test_size", required = False, type = float, default = 0.5, 
                    help = "int or float:  A proportion of the data the model to be tested on")
    ap.add_argument("-r", "--random_state", required = False, default = None, type = int, 
                    help = "int: a random seed")
    ap.add_argument("-sm","--save_metrics", required = False, action = "store_true", 
                    help = "bool: whether to save the metrics")
    ap.add_argument("-mf", "--metrics_filename", required = False, default =  "neural_network_metrics.txt",type = str, 
                    help = "the filename of the metrics with the .txt ending")
    ap.add_argument("-e", "--epochs", required = False, default = 80, type = int, 
                    help = "int: A number of epochs")
    ap.add_argument("-hl", "--hidden_layers", required = False, type = int, default = [32,16], nargs = "+", )
    
    
    # parse arguments, parse the argumets and returns them as a list of arguments/variables
    args = vars(ap.parse_args())
    print(args)
    
   # instead of listing all comman line interface arguments separately, we can list all of them at once with **args
    main(**args)
