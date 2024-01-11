##BY RIAN O DONNELL####
##BRAIN SCAN RECOGNITION 2023##

import os
import numpy as np

import cv2
import random
import math
from numpy.core.fromnumeric import mean
import json
from numpy import savetxt
from numpy import asarray
import csv
storage = open("storage.txt", "w")


#take the input and caompare it to the targets 1e-7 -1e-7 y_pred_clipped[range(lenght of samples, targets)]


class reLU():

  def forward(self, inputs):
    self.output = np.maximum(0, inputs)
    self.inputs = inputs

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0  # if negative they are zero


class Layer_dense():

  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

    self.biases = np.zeros((1, n_neurons))

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  def backward(self, dvalues):
    #Gradients on params
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    #gradients for values
    self.dinputs = np.dot(dvalues, self.weights.T)


class softmax():

  def forward(self, input):
    exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
    #eliminates dedad neurons and very large neurons
    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = norm_values

  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)  #similar array but not initliazes
    for index, (single_ouput, single_dvalues) in enumerate(
        zip(self.output, dvalues)):  #zips output and prevoious derivatives
      single_output = single_output.reshape(-1, 1)

      jacobian_matrix = np.diagflat(single_output) - np.dot(
          single_output, single_output.T)


class loss():

  def calculate(self, outputs, targets):

    sample_loss = self.forward(outputs, targets)

    self.data_loss = mean(sample_loss)

    return self.data_loss


class crossentropy(loss):

  def forward(self, y_pred, y_true):
    samples = len(y_pred)

    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  #clippinf numbers close to zero

    if len(y_true.shape) == 1:

      correct_confidences = y_pred_clipped[range(samples), y_true]
    elif len(y_true.shape) == 2:
      correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)  #if

    neg_log = -np.log(correct_confidences)

    return neg_log

  def backward(self, dvalues, y_true):
    samples = len(dvalues)  # number of samples
    labels = len(dvalues[0])  #counting them ]

    if len(y_true.shape) == 1:
      y_true = np.eye
    self.dinputs = dvalues.copy()
    self.dinputs[range(samples), y_true] - self.dinputs  #calculate gradients
    self.dinputs = self.dinputs / samples  #cleansing data


class Activation_softmax_crossentropy():

  def __init__(self):
    self.activation = softmax()
    self.loss = crossentropy()

  def forward(self, inputs, y_true):
    #gte inputs from outer layer
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)

  def backward(self, dvalues, y_true):
    samples = len(dvalues)

    #if the labels are one hot encoded turn them into discrete values

    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)

    #copy derivatives so we dont modify
    self.dinputs = dvalues.copy()
    #gradient
    self.dinputs[range(samples), y_true] -= 1
    self.dinputs = self.dinputs / samples


class optimizer:

  def __init__(self,learning_rate=1,decay=0, epsilon=1e-7,beta1=0.9,beta2=0.999):
    self.iterations = 0
    self.learning_rate = learning_rate
    self.current_learningrate = learning_rate
    self.decay = decay
    self.beta1 = beta1
    self.beta2 = beta2

    self.epsilon = epsilon

  def learning_rate_decay(self, step):
    self.current_learningrate = self.learning_rate * (1. / (1 + self.decay * step))
    return self.current_learningrate

  def update_params(self, layer):
      #if a leyr doesnt have momentum array add it
      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = np.zeros_like(layer.weights)  #array the same same as weights but filled iwth zeros
        layer.bias_momentums = np.zeros_like(layer.biases)
        layer.weight_cache = np.zeros_like(layer.weights)
        layer.bias_cache = np.zeros_like(layer.biases)
        #
      #entire optimizing calculation

      #AdaM optimizer
      layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights

      layer.bias_momentums = self.beta1 * layer.bias_momentums + (1-self.beta1) * layer.dbiases

            #iteration starts at zero this corrects it
      weight_momentums_corrected = layer.weight_momentums / (1-self.beta1 ** (self.iterations + 1))
      bias_momentums_corrected = layer.bias_momentums / (1-self.beta1 ** (self.iterations + 1))
      #stores as cache to add onto momentums next time
      layer.weight_cache = self.beta2 * layer.weight_cache + (1- self.beta2) * layer.dweights**2
      layer.bias_cache = self.beta2* layer.bias_cache + (1-self.beta2) * layer.dbiases**2
      #take out the 0 iteration error
      self.wcache_updates = layer.weight_cache / (1-self.beta2 **(self.iterations+1))
      self.dcache_updates = layer.bias_cache / (1-self.beta2 **(self.iterations+1))


      layer.weights += -self.current_learningrate * weight_momentums_corrected / (np.sqrt(self.wcache_updates) + self.epsilon)

      layer.biases += -self.current_learningrate * bias_momentums_corrected / (np.sqrt(self.dcache_updates) + self.epsilon)
  def iterations_update(self):
    self.iterations += 1
with open('BTDS/data.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    y = np.array(data, dtype=int)
y = (y.reshape(1,-1))
y = (y.reshape(1,-1))
print(y.shape)
print(y[0])
y = y[0]
def runmodel(w1, b1, w2, b2,w3, b3, softmax):
            softmax = softmax
            image_data = cv2.imread('samples/image.png', cv2.IMREAD_GRAYSCALE)
            image_data = cv2.resize(image_data, (50,50))
            dense1.weights = w1
            dense2.weights = w2
            dense1.biases = b1
            dense2.biases = b2
            dense3.weights = w3
            dense3.biases = b3
            X2 = image_data

            np.set_printoptions(linewidth=200)

            X2 = 255 - X2

            #pos = int(input("Select from where in the testing dataset you want to pick your number (max 10,000): "))
            print("Selected number:")
             #inverts colours






            print(X2)
            X2 = (X2.reshape(1, -1))

            print("----")


            dense1.forward(X2)
             # inverts the inverted colour prblem
            activation.forward(dense1.output)  # ReLU
            output = ("No brain tumor detected", "Brain Tumor detected")
            dense2.forward(activation.output)
            activation.forward(dense2.output)
            dense3.forward(activation.output)
            softmax.forward(dense3.output)
            output1 = softmax.output
            max1x = max(output1[0])
            #predicts using softmax prediction output
            print("Prediction: ")
            #Debugging options: Shows how confident teh model was
            #print(output1)
            #print(sum(output1[0]))
            maxnum = (next(i for i, x in enumerate(output1[0]) if x == max1x))

            print(output[maxnum])


samples = ("/BTDS/BrainTumor")
images = np.array([])
import sys
from matplotlib import pyplot as plt
def load_images_from_folder(folder):
    global images

    count = 1
    for filename in os.listdir(folder):


        filename = (folder + '\\' + str(filename))

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200,200))
        np.set_printoptions(linewidth=200)
        img = 255 - img
        progress = ((count / 4600) * 100)

        img = (img.reshape(1, -1))

        if count == 1:


            images = img
            print(images)


        elif img is not None:



            images = np.append(images, img, axis=0)
            print("Progress: ", round(progress, 2), "%")


        count = count + 1

    print(images)
    return images

alreadyloadedimages = False
folder1 = (r"C:\Users\conor\Downloads\Brainscans")
import os
for filename in os.listdir(folder1):

    file_name, file_extension = os.path.splitext(filename)

    if file_extension == ".csv":
        alreadyloadedimages = True
import time
if alreadyloadedimages == False:
    print("Image preprocessing not detected")
    print("Building file in allocated directory")
    time.sleep(3)
    load_images_from_folder(samples)
    os.system('cls')
    print("Downloading to file")
    savetxt('X.csv',images, delimiter=',' )

with open('X.csv', 'r') as f:
    print("Opening file")
    reader = csv.reader(f)
    data = list(reader)
    X = np.array(data, dtype=float)


print(X.shape)
print(y.shape)

np.set_printoptions(linewidth = 150)

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X.reshape(X.shape[0], -1) #
print("============================================================")
option = input("Commands: T - Train | R - Run model | H - Help: ")
dense1 = Layer_dense(40000,40000)
dense2 = Layer_dense(40000,40000)
activation = reLU()
dense3 = Layer_dense(40000, 2)
loss_activation = Activation_softmax_crossentropy()
softmax1 = softmax()

optimizer = optimizer(learning_rate=0.01, decay=1e-3)

X = X[keys]
y = y[keys]
print(X[1])
print(y[1])

if option.upper() == "T":

    batch_size = int(input("Batch size for training / max 4600: "))
    X = X[:batch_size]
    y = y[:batch_size]


    #(X1, y1), (X_test, y_test) = mnist.load_data()


    #the mean of corretc predicitons

    lowest_loss = 9000
    EPOCHS = 10001
    dense1weights = []
    dense2weights = []
    dense1biases = []
    dense2biases = []

    for i in range(EPOCHS):
      dense1.forward(X)

      activation.forward(dense1.output)  # ReLU

      dense2.forward(activation.output)
      activation.forward(dense2.output)
      dense3.forward(activation.output)
      loss = loss_activation.forward(dense3.output, y)  # y is the predicted valeus
      predictions = np.argmax(loss_activation.output, axis=1)
      if len(y.shape) == 2:  # if its 2 dimensions
        y = np.argmax(y, axis=1)
      accuracy = np.mean(predictions == y)

      loss_activation.backward(loss_activation.output, y)  # backward pass of loss activation and predicted values
      dense3.backward(loss_activation.dinputs)
      activation.backward(dense3.dinputs)
      dense2.backward(activation.dinputs)  # derivatives from loss activation
      activation.backward(dense2.dinputs)
      dense1.backward(activation.dinputs)

      decay = optimizer.learning_rate_decay(i)
      optimizer.update_params(dense1)
      optimizer.update_params(dense2)
      optimizer.update_params(dense3)
      optimizer.iterations_update()
      learningrate = decay
      if i % 100 == 0:
        print("iteration: ", i)
        print("accuracy: ", accuracy)
        print("loss: ", loss)
        print("learning_rate: ", decay)
      if i % 100 and i > 300:
          choice = input("Do you want to end the program y/n")
      if choice == "y":
          print("Accuracy: ", accuracy)
          choice = input("Do you want to save the parameters? y/n")
          if choice == "y":
            savetxt('weights1.csv',dense1.weights, delimiter=',' )
            savetxt('biases1.csv',dense1.biases, delimiter=',' )
            savetxt('weights2.csv',dense2.weights, delimiter=',' )
            savetxt('biases2.csv',dense2.biases, delimiter=',' )
            savetxt('weights3.csv',dense3.weights, delimiter=',' )
            savetxt('biases3.csv',dense3.biases, delimiter=',' )
            w1 = dense1.weights
            b1 = dense1.biases
            w2 = dense2.weights
            b2 = dense2.biases
            w3 = dense3.weights
            b3 = dense3.biases

            break
#put your own handwritten number
elif option.upper() == "R":
    with open('weights1.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        w1 = np.array(data, dtype=float)
    with open('weights2.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            w2 = np.array(data, dtype=float)
    with open('biases1.csv', 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
                b1 = np.array(data, dtype=float)
    with open('biases2.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            b2 = np.array(data, dtype=float)
    with open('biases3.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            b3 = np.array(data, dtype=float)
    with open('weights3.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            w3 = np.array(data, dtype=float)
    runmodel(w1, b1, w2, b2,w3, b3, softmax1)

else:
    print("=========================== HELP ===========================")
    print("- The model is trained on the MNIST handwriting dataset")
    print("- To add a custom image take a square picture of equal pixels ")
    print("Then add it to the samples folder and call it image.png")
    print("- From Rian")



file3 = 'samples/t10k-images.idx3-ubyte'
file4 = 'samples/t10k-labels.idx1-ubyte'




'''output1 = (max(softmax.output))
print("predicted value: ")
print((softmax.output).index(output1))
'''
    #store
  #determines accuary from softmax fucntions
'''X1 = idx2numpy.convert_from_file(file)
y1 = idx2numpy.convert_from_file(file2)
np.set_printoptions(linewidth = 150)

keys = np.array(range(X1.shape[0]))
np.random.shuffle(keys)
X1 = X1.reshape(X1.shape[0], -1) #
print(y1.shape)
X1 = X1[:10000]
X1 = X1[keys]
y1 = y1[keys]
y1 = y1[:10000]
print(y1[:15])
'''
