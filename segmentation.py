import numpy as np
from tensorflow.keras.models import load_model
import cv2


'''
Returns the (x, y) coordinate of the centroid of the needle in an ultrasound image
PARAMS:
- x: A grayscale image of shape (w, h, 1), with pixel intensities normalized to [0.0, 1.0]
RETURNS: The predicted centroid coordinate, of shape (2,)
'''
def segment_image(x):

    # Get original image dimensions
    w = x.shape[0]
    h = x.shape[1]

    # Resize image to correct input size for neural network
    w_resized = model.layers[0].output_shape[1]
    h_resized = model.layers[0].output_shape[2]
    x = np.expand_dims(cv2.resize(x, (w_resized, h_resized)), axis=3)

    # Predict coordinates. Result is in range [-1, 1]
    y = model.predict(np.expand_dims(x, axis=0)).T

    # Scale predicted coordinates to pixel coordinate
    y[0] = int((y[0] + 1.0) / 2.0 * w)
    y[1] = int((y[1] + 1.0) / 2.0 * h)

    # Reshape y to a row vector
    y = np.squeeze(y.T, axis=0)

    return y


# Load the neural network model
model = load_model('model_best.h5')