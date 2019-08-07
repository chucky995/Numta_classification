import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from  sklearn.model_selection import train_test_split

print("GPU Available: ", tf.test.is_gpu_available())   ##### Checks for gpu availability, use gpu if possible

PATH = '/home/anik/Desktop/numta/training-a/'  ### Enter own path here(directory that contains all the images
images = sorted(glob(os.path.join(PATH, "*.png")))
labels = pd.read_csv('/home/anik/Desktop/numta/training_a_labels.csv', header =None)  ### create a separate csv file with only the labels and read them
# print(labels[0].value_counts())

X = []
IMG_SIZE = 90   ##### You can choose smaller but not so much that the writing isn't distinguishable anymore
def create_training_data():

    for image in images:
        image_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
        X.append(image_array)
#   plt.imshow(image_array, cmap = 'gray')
#   plt.show()
#   break
    return X
X = create_training_data()
print("Images loaded.")
plt.imshow(X[2])
plt.show()
y = np.array(labels)
print("Label for the image: ", y[2])
X= np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Images resized")

###################################################################################################

#####    Use if using jupyter notebook, helps avoid loading the images again and again

"""
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close


X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))

"""
###################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Train test split done.")
X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)
print("Training and tetsing data normalisation done.")

##  Replace the following with your favourite model to train


model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), input_shape = X_train.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy',
            metrics  = ['accuracy'])

print("Starting training")
model.fit(X_train, y_train, epochs = 20, batch_size = 128)

val_loss, val_acc = model.evaluate(X_test, y_test)
print("Validation Loss: ", val_loss, "Validation Accuracy:", val_acc)





