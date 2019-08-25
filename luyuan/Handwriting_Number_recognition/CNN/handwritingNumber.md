# HandWriting Number Recognition (CNN) by Pytorch
created by Lu Yuan, August 24th, 2019

## Environment
1. Python 3.6
2. Pytorch 1.0, cpu version

## Main Steps
1. Download dataset from MNIST
2. Use matplotlib to show an example 
![screen ](screencut/1.png)
3. Define class CNN(convolutional neural network), which contains conv1, max pooling1, conv2, max pooling2, and fully connected layer
4. train

## Super parameters
1. epoch (times of training data) is 1, for saving resource
2. mini batch size is 50
3. learning rate is 0.001

## Result 
1. The accuracy arrives 98% finally and the 10 test example are all predicted correctly
![screen ](screencut/2.png)
