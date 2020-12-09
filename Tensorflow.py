#
# https://www.tensorflow.org/tutorials/keras/classification
#

#Tensorflow Library
import tensorflow as tf

#Helper Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

################
#VERIFY DETAILS#
################

#Print Tensorflow Version
print(tf.__version__)


#Use fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

#Training Variables
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Define Class Names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Check Data
print(train_images.shape)

#List No. of Labels
print(len(train_labels))

#Find what integer is each leabel
print(train_labels)

#No. of images in the test set for this program alone with an x28 resolution.
print(test_images.shape)

#No. of image labels
print(len(test_labels))

#Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
#To do so, divide the values by 255. 
#It's important that the training set and the testing set be preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

#Verify that the data is correct by displaying the first 25 images
#in training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#######################
#SETTING UP THE LAYERS#
#######################

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#####################
#COMPILING THE MODEL#
#####################

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####################
#TRAINING THE MODEL#
####################

model.fit(train_images, train_labels, epochs=10)

#########################
#EVALUATING THE ACCURACY#
#########################

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

####################
#MAKING PREDICTIONS#
####################

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions)
predictions[0]

#A prediction is an array of 10 numbers. 
#They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. 
#See which label has the highest confidence value:
np.argmax(predictions[0])
print(np.argmax(predictions[0]))

#Examine the test labels to see if the prediction is correct
test_labels[0]
print(test_labels[0])

#Lets graph the results to look at the full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#######################
#VERYFYING PREDICTIONS#
#######################

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#######################
#USE THE TRAINED MODEL#
#######################

#Finally , lets use the trained model to make a prediction about a single image

#Grab an image from the test dataset
img = test_images[12]
print(img.shape)

#Keras models are optimised to make predctions on a batch.
#Even though you're using a single image, 
#you need to add it to a list
img = (np.expand_dims(img,0))
print(img.shape)

#Now lets make the module predict the correct label for this image
predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
imgplot = plt.imshow(test_images[12])
plt.show()

np.argmax(predictions_single[0])
print(np.argmax(predictions_single[0]))