import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import time
import datetime



'''
# Define the architecture of the model for Tensorflow compilation
'''
def define_model(input_shape):

    # Define input placeholder as tensor with shape input_shape
    X_input = Input(input_shape)

    # Series of [Convolution -> LeakyReLU -> MaxPool] layers applied to X
    X = Conv2D(16, (3, 3), strides=(1, 1), activity_regularizer=l2(0.00001), name='conv0')(X_input)
    X = LeakyReLU()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool0')(X)
    X = Conv2D(32, (2, 2), strides=(1, 1), activity_regularizer=l2(0.00001), name='conv1')(X)
    X = LeakyReLU()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1')(X)
    X = Conv2D(64, (2, 2), strides=(1, 1), activity_regularizer=l2(0.00001), name='conv2')(X)
    X = LeakyReLU()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool2')(X)
    X = Conv2D(128, (2, 2), strides=(1, 1), activity_regularizer=l2(0.00001), name='conv3')(X)
    X = LeakyReLU()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3')(X)
    X = Conv2D(256, (2, 2), strides=(1, 1), activity_regularizer=l2(0.00001), name='conv4')(X)
    X = LeakyReLU()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool4')(X)

    # Flatten all nodes and pass through a series of [Fully Connected -> LeakyReLU] layers
    X = Flatten()(X)
    X = Dense(1024, name='fc0')(X)
    X = LeakyReLU()(X)
    X = Dense(128, name='fc1')(X)
    X = LeakyReLU()(X)
    X = Dense(16, name='fc2')(X)
    X = LeakyReLU()(X)
    Y = Dense(2, activation='linear', name='output')(X) # Output passes through linear activation function

    model = Model(inputs=X_input, outputs=Y, name='CoordinateModel')
    print(model.summary())
    return model


# Load training data. Data can be generated using generate_data.py
imgs = np.load('images.npy')
itns = np.load('intersections.npy')

# Get indices of images that contain the fiducial's centroid
indices = np.argwhere(itns[:,2] == 1.0).flatten()
X = np.take(imgs, indices, axis=0)
Y = np.take(itns, indices, axis=0)
Y = np.delete(Y, 2, 1) # Delete 3rd column containing flag for intersection
Y = 2.0*Y - 1.0 # Normalize intersections to [-1,1]

n = X.shape[0] # The number of training examples

# Create random ordering of indices for assignment to training/test sets
np.random.seed(int(time.time())%1000)
random_idx = np.random.randint(0,n,n+1)
stop_idx = int(0.9*n) # Training set will be 90% of the data and preliminary test set will be 10% of the data.
# For now, use part of training set for testing. Remember to test on a separate dataset later.

# Partition data into training and test sets
X_train = np.zeros((stop_idx, X.shape[1], X.shape[2], X.shape[3]))
X_test = np.zeros(((n - stop_idx), X.shape[1], X.shape[2], X.shape[3]))
Y_train = np.zeros((stop_idx, Y.shape[1]))
Y_test = np.zeros(((n - stop_idx), Y.shape[1]))
for i in range(0,stop_idx):
    X_train[i] = X[random_idx[i]]
    Y_train[i] = Y[random_idx[i]]
for i in range(stop_idx,n):
    X_test[i-stop_idx] = X[random_idx[i]]
    Y_test[i-stop_idx] = Y[random_idx[i]]

# Define early stopping callback
patience = 15
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=patience, verbose=0, mode='auto')

# Define model checkpoint callback
model_checkpoint = ModelCheckpoint('model_best.h5', monitor='loss',verbose=1, save_best_only=True)

# Define learning rate decay callback
def scheduler(epoch):
    lr = 0.0001
    if epoch < 100:
        lr = 0.0001
    elif epoch < 200:
        lr = 0.00007
    return lr
change_lr = LearningRateScheduler(scheduler)

# Define TensorBoard callback
LOG_DIRECTORY_ROOT = ''
now = datetime.date.today().strftime("%Y%m%d%H%M%S")
log_dir = "".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir='./logs')
callbacks = [early_stopping, tensorboard, change_lr, model_checkpoint]

# Load validation set
X_val = np.load('images_val.npy')
Y_val = np.load('intersections_val.npy')
Y_val = np.delete(Y_val, 2, 1) # Delete 3rd column
Y_val = 2*Y_val - 1.0 # Normalize intersections to [-1, 1]

# Define neural network model and its optimizer
model = define_model((X.shape[1], X.shape[2], X.shape[3]))  # Create the NN model

# Set optimizer and loss function for gradient descent. Print accuracy metrics.
model.compile(optimizer='adam', loss="mean_absolute_error", metrics=["accuracy"])

# Train the model
model.fit(x=X_train, y=Y_train, epochs=100, batch_size=64, callbacks=callbacks, validation_data=(X_val, Y_val))

# Evaluate model on the preliminary test set
preds = model.evaluate(x=X_test, y=Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

# Save the model in Hierarchical Data Format
model.save('model.h5')
