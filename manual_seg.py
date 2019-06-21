import matplotlib.pyplot as plt
import numpy as np

# Global variable declarations
global X, Y, idx, w

# Load images and corresponding labels
X = np.load('images.npy')
Y = np.load('intersections.npy')

idx = 0
w = X.shape[1]
h = X.shape[2]
del_idxs = []

'''
Event handler for user interaction with the keyboard. Used for traversing, saving, and deleting images and intersections
PARAMS:
- event: the event triggered by pressing a button on the keyboard
'''
def key_pressed(event):
    if event.key == 'c':
        globals()['idx'] = (idx + 1)%X.shape[0] # Progress to the next image in the sequence
        display_image()
    elif event.key == 'z':
        globals()['idx'] = (idx - 1)%X.shape[0] # Return to the previous image in the sequence
        display_image()
    elif event.key == 'd':
        display_image()
        globals()['del_idxs'].append(idx) # Add the index of the currently displayed image to the delete list
        print(del_idxs)
    elif event.key == 'x':
        np.save("images_val", X) # Save images to a .npy file
        np.save("intersections_val", Y) # Save intersections to a .npy file
        print("Data saved")
    elif event.key == 'e':
        globals()['del_idxs'] = list(set(del_idxs)) # Remove duplicate indices from the delete list
        print(del_idxs)
        globals()['X'] = np.delete(X, del_idxs, 0) # Delete images at the indices in the delete list
        globals()['Y'] = np.delete(Y, del_idxs, 0) # Delete intersections at the indices in the delete list
        globals()['idx'] = 0
        globals()['del_idxs'] = [] # Clear the delete list
        display_image()

'''
Event handler for user intersection with the mouse. Used for updating the segmentation of an image.
PARAMS:
- event: the event triggered by pressing the mouse
'''
def mouse_pressed(event):
    # Update the coordinates of the needle intersection according to where the user clicked on the image.
    Y[idx][0] = event.ydata/w
    Y[idx][1] = event.xdata/h

    print(event.xdata)
    print(event.ydata)
    display_image()

'''
Displays the image at idx to the user, along with its corresponding label for needle intersection
'''
def display_image():
    sample_img = np.squeeze(X[idx], axis=2)
    plt.clf()
    plt.imshow(sample_img, cmap="gray")  # Display an image
    plt.scatter(Y[idx][1] * h, Y[idx][0] * w, color='r', s=5) # Plot the needle intersection label
    plt.title(str(idx) + ' / ' + str(Y.shape[0] - 1)) # Display the index of the current image to the user
    fig.canvas.draw()
    plt.show()


# Set up event handlers for keyboard and mouse press events
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', key_pressed)
fig.canvas.mpl_connect('button_press_event', mouse_pressed)
display_image() # Display the first image to the user



