# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.model1 import Model1
from pyimagesearch.model2 import Model2
from pyimagesearch.model3 import Model3
import matplotlib.pyplot as plt
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
# python train.py --dataset dataset --model fashion2.model --labelbin mlb.pickle
EPOCHS = 6
EPOCHS2 = 1
EPOCHS3 = 1
INIT_LR = 1e-3 #default for Adam optimizer
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

data2 = []
labels2 = []

data3 = []
labels3 = []

# loop over the input images
for index, imagePath in enumerate(imagePaths):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	
		
	#if(index%3 == 0):
	data.append(image)
	#if(index%3 == 1):
	data2.append(image)
	#if(index%3 == 2):
	data3.append(image)

	l = label = imagePath.split(os.path.sep)[-2].split("_")
	#if(index%3 == 0):
	labels.append(label)
	#if(index%3 == 1):
	labels2.append(label)
	#if(index%3 == 2):
	labels3.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

data2 = np.array(data2, dtype="float") / 255.0
labels2 = np.array(labels2)

data3 = np.array(data3, dtype="float") / 255.0
labels3 = np.array(labels3)

print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")


mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)


labels2 = mlb.fit_transform(labels2)
labels3 = mlb.fit_transform(labels3)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))
# for (i, label2) in enumerate(mlb.classes_):
# 	print("{}. {}".format(i + 1, label2))
# for (i, label3) in enumerate(mlb.classes_):
# 	print("{}. {}".format(i + 1, label3))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
(trainX2, testX2, trainY2, testY2) = train_test_split(data2, labels2, test_size=0.2, random_state=42)

# print(trainX, testX, trainY, testY)
# print(trainX2, testX2, trainY2, testY2)
# print(trainX3, testX3, trainY3, testY3)

(trainX3, testX3, trainY3, testY3) = train_test_split(data3, labels3, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = Model1.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
model2 = Model2.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
model3 = Model3.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt2 = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS2)
opt3 = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS3)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


model2.compile(loss="binary_crossentropy", optimizer=opt2, metrics=["accuracy"])

model3.compile(loss="binary_crossentropy", optimizer=opt3, metrics=["accuracy"])

# train the network
print("[INFO] training network 1...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] training network 2...")
H2 = model2.fit_generator(
	aug.flow(trainX2, trainY2, batch_size=BS),
	validation_data=(testX2, testY2),
	steps_per_epoch=len(trainX2) // BS,
	epochs=EPOCHS2, verbose=1)

print("[INFO] training network 3...")
H3 = model3.fit_generator(
	aug.flow(trainX3, trainY3, batch_size=BS),
	validation_data=(testX3, testY3),
	steps_per_epoch=len(trainX3) // BS,
	epochs=EPOCHS3, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"]+"1")
model2.save(args["model"]+"2")
model3.save(args["model"]+"3")

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()


preds = model.predict(testX)
preds2 = model.predict(testX2)
preds3 = model.predict(testX3)

labelNames = ["black", "blue", "dress", "jeans", "red", "shirt"]

proba = preds[0] 
proba2 = preds2[0] 
proba3 = preds3[0] 

mlb = pickle.loads(open(args["labelbin"], "rb").read())
# show the probabilities for each of the individual labels
#python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg
# python classify.py --model fashionSet.model2 --labelbin mlb.pickle --image examples/example_01.jpg
print("Skuteczność klasyfikacji pojedyńczych atrybutów")
print("Skuteczność dla modelu 1")
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))

print("Skuteczność dla modelu 2")
for (label, p) in zip(mlb.classes_, proba2):
	print("{}: {:.2f}%".format(label, p * 100))

print("Skuteczność dla modelu 3")
for (label, p) in zip(mlb.classes_, proba3):
	print("{}: {:.2f}%".format(label, p * 100))

# print(preds)

# testY = np_utils.to_categorical(testY, 6)
# preds = np_utils.to_categorical(preds)

# print(testY.argmax(axis=1))

# print(preds.argmax(axis=1))

# # show a nicely formatted classification report
# print("[INFO] evaluating network 1...")
# print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
# 	target_names=labelNames))

# print("[INFO] evaluating network 2...")
# print(classification_report(testY2.argmax(axis=1), preds2.argmax(axis=1),
# 	target_names=labelNames))

# print("[INFO] evaluating network 3...")
# print(classification_report(testY3.argmax(axis=1), preds3.argmax(axis=1),
# 	target_names=labelNames))

# plot the training loss and accuracy

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("1"+args["plot"])

plt.style.use("ggplot")
plt.figure()
N = EPOCHS2
plt.plot(np.arange(0, N), H2.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H2.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H2.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H2.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("2"+args["plot"])

plt.style.use("ggplot")
plt.figure()
N = EPOCHS3
plt.plot(np.arange(0, N), H3.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H3.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H3.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H3.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("3"+args["plot"])


#sudo docker run --name tensorflow -v ${PWD}/notebook:/notebook -p 8888:8888 -p 6006:6006 drunkar/anaconda-tensorflow-gpu /bin/bash