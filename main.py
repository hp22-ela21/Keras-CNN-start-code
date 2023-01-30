################################################################################
# main.py: Demonstration of convoluational neural network implementation
#          by using the Keras library.
################################################################################
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cnn


def main():
   """
   main: Creates convolutional neural network consisting of the following layers:
   
         1. A convolutional layer with image size = 64x64, kernel size = 3x3 and strides = 1.
         Zero padding is used to keep the image size the same after feature extraction.
         Input shape of the network is set to 32x32x3 for color images.

         2. A pooling layer with pool size = 3x3 and strides = 1.

         3. A zero padded convolutional layer with image size = 32x32, 
            kernel_size = 2x2 and strides = 1.

         4. A pooling layer with pool size = 2x2 and strides = 1.

         5. A zero padded convolutional layer with image size = 16x16, 
            kernel_size = 2x2 and strides = 1.

         6. A pooling layer with pool size = 2x2 and strides = 1.

         7. A flatten layer to convert the extracted image from 2D to 1D before
            the following dense layers.

         8. A dense layer consisting of 50 nodes, with ReLU used as activation function. 

         9. A dense layer consisting of 10 nodes, with sigmoid used as activation function.

         Information summary about the network is printed in the terminal by 
         calling the model's summary method. The model is then trained during 
         10 000 epochs by calling the train function in the cnn module. 
   """
   model = Sequential()
   model.add(Conv2D(64, 3, strides = 1, padding = "same", activation = "relu", input_shape = (32, 32, 3)))
   model.summary()
   return 

################################################################################
# Calls the main function to start the program if this is the startup file.
################################################################################
if __name__ == "__main__":
   main()

