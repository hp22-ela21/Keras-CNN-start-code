################################################################################
# cnn.py: Module for simple implementation of convolutional neural networks
#         (CNN) in Python. The CIFAR-10 dataset is used to train the networks.
################################################################################
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
import numpy as np

class dataset:
   """
   dataset: Dataset containing 60 000 32x32 pictures of cars, cats, dogs, birds etc.
            The pictures are loaded from the CIFAR-10 dataset, which is normally
            used to train convolutional neural networks.
   """

   def __init__(self):
      """
      __init__: Loads dataset with pictures copied from CIFAR-10 dataset.
      """
      self.xtrain = []
      self.ytrain = []
      self.xvalid = []
      self.yvalid = []
      self.__load()
      return

   def __load(self):
      """
      __load: Loads the CIFAR-10 dataset and copies into current dataset.
      """
      (self.xtrain, self.ytrain), (self.xtest, self.ytest) = cifar10.load_data()
      categories = len(np.unique(self.ytrain))

      self.xtrain = self.xtrain.astype("float32") / 255.0
      self.xtest = self.xtest.astype("float32") / 255.0

      self.ytrain = to_categorical(self.ytrain, num_classes = categories)
      self.ytest = to_categorical(self.ytest, num_classes = categories)

      self.xtrain, self.xvalid = self.xtrain[5000 : ], self.xtrain[ : 5000]
      self.ytrain, self.yvalid = self.ytrain[5000 : ], self.ytrain[ : 5000]
      return

def train(model, num_epochs = 10000):
   """
   train: Trains referenced network during specified number of epochs and stores the best 
            parameters in a file named "model.weights.best.hdf5". Efter training the 
            best parameters are load from the file and used in the model.

            - model     : Sequential neural network model.
            - num_epochs: Number of training epochs to perform (default = 10000).
   """
   data = dataset()
   filepath = "model.weights.best.hdf5"
   checkpoint = ModelCheckpoint(filepath = filepath, verbose = 1, save_best_only = True)
   model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])
   hist = model.fit(data.xtrain, data.ytrain, batch_size = 32, epochs = num_epochs, 
                           validation_data = (data.xvalid, data.yvalid), 
                           callbacks = [checkpoint], verbose = 2, shuffle = True)
   model.load_weights(filepath)
   return
