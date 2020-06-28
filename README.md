# Image Classification using Logistic Regression in PyTorch

In this tutorial, we'll use our existing knowledge of PyTorch and linear regression to solve a very different kind of problem: image classification. We'll use the famous MNIST Handwritten Digits Database as our training dataset. It consists of 28px by 28px grayscale images of handwritten digits (0 to 9), along with labels for each image indicating which digit it represents. Here are some sample images from the dataset:



## Exploring the Data
We begin by importing torch and torchvision. torchvision contains some utilities for working with image data. It also contains helper classes to automatically download and import popular datasets like MNIST.
