import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from segmentation import segment_image


'''
An event handler for keyboard interaction. Used for browsing images and predicted centroids.
PARAMS:
- event: the keyboard press event
'''
def key_pressed(event):
    if event.key == 'c':
        globals()['idx'] = (idx + 1)%X.shape[0] # Move to the next image in the list
        make_prediction(idx) # Predict the centroid of the needle in the image
    elif event.key == 'z':
        globals()['idx'] = (idx - 1)%X.shape[0] # Move to the previous image in the list
        make_prediction(idx) # Predict the centroid of the needle in the image

'''
RMSE loss function
PARAMS:
- y_true: Data labels
- y_pred: Predicted outputs of neural network
RETURNS: the RMSE as a float value
'''
def rmse(y_true, y_pred):
    return tf.math.sqrt(tf.losses.mean_squared_error(y_true, y_pred))


'''
Predicts the centroid of the needle in an image using the neural network model
PARAMS:
- x: An ultrasound image
RETURNS: the (x, y) prediction for the centroid of the needle
'''
def predict_centroid(x):
    y = model.predict(np.expand_dims(x, axis=0)).T
    return y


'''
Display an image on a plot and the coordinate corresponding to the centroid of the needle in the image.
'''
def display_image(idx, p):
    p = (p + 1.0) / 2.0
    img = np.squeeze(X[idx], axis=2) # Select image from the data set
    y = Y[idx] # Select corresponding label from data set
    plt.clf()
    plt.imshow(img, cmap="gray")  # Display the image
    plt.scatter(y[1] * X.shape[2], y[0] * X.shape[1], color='r', s=5)
    if p[0] != -1:
        p[0] = p[0] * X.shape[1]
        p[1] = p[1] * X.shape[2]
    plt.scatter(p[1], p[0], color='b', s=5) # Plot the centroid point
    plt.title(str(idx) + ' / ' + str(Y.shape[0] - 1))
    fig.canvas.draw()
    plt.show()


'''
Predict the centroid for a single image and display the result.
'''
def make_prediction(idx):
    p = predict_centroid(X[idx])
    display_image(idx, p)


'''
Evaluate the model's performance on the current data set, and print the results.
'''
def test_whole_set():
    coords = np.delete(Y, 2, 1)
    coords = 2.0 * coords - 1.0
    model.compile(optimizer='adam', loss=rmse, metrics=["accuracy"])
    preds = model.evaluate(x=X, y=coords)  # Evaluate model's performance on the test set
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))


'''
Predict the coordinates of the centroid of the needle intersection for all images in the currently loaded dataset
RETURNS: A list of coordinates
'''
def predict_whole_set():
    return xy_model.predict(X)

def rmse_in_pixels():
    Y_pred = predict_whole_set() # Predict centroid for every image in the data set. Results are in range [-1, 1]
    Y_pred = (Y_pred + 1.0) / 2.0 # Normalize to [0, 1]
    Y_true = np.delete(Y, 2, 1)

    # Dimensions of images used in the experiment
    w = 356
    h = 589

    # Scale to image dimensions
    Y_pred[:, 0] = Y_pred[:, 0] * w
    Y_pred[:, 1] = Y_pred[:, 1] * h
    Y_true[:, 0] = Y_true[:, 0] * w
    Y_true[:, 1] = Y_true[:, 1] * h

    # Calculate RMSE in pixels
    sum = 0
    n = Y.shape[0]
    for i in range(0, n):
        sum += np.square(Y_true[i] - Y_pred[i])
    rmse = np.sqrt(sum / n)
    return rmse

'''
Get mean absolute error for the entire data set
'''
def mae_in_pixels():
    Y_pred = predict_whole_set()  # Predict centroid for every image in the data set. Results are in range [-1, 1]
    Y_pred = (Y_pred + 1.0) / 2.0  # Normalize to [0, 1]
    Y_true = np.delete(Y, 2, 1)

    # Dimensions of images used in the experiment
    w = 356
    h = 589

    # Scale to image dimensions
    Y_pred[:, 0] = Y_pred[:, 0] * w
    Y_pred[:, 1] = Y_pred[:, 1] * h
    Y_true[:, 0] = Y_true[:, 0] * w
    Y_true[:, 1] = Y_true[:, 1] * h

    # Calculate MAE in pixels
    sum = 0
    n = Y.shape[0]
    for i in range(0, n):
        sum += np.abs(Y_true[i] - Y_pred[i])
    mae = sum / n
    return mae


# Load a data set
X = np.load('images_test.npy')
Y = np.load('intersections_test.npy')
idx = 0

# Laod a model
model = load_model('model_best.h5')

# Make a prediction for the first image in the data set and display results on a plot
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', key_pressed)
make_prediction(idx)
