# # HandWriting Number Encode by Pytorch
created by Lu Yuan, August 26th, 2019

## Environment
1. Python 3.6
2. Pytorch 1.0, cpu version

## Main Steps
1. Download dataset from MNIST
2. Define class AutoEncoder, which contains several Linear Squence layers 
3. train

## Hyper parameters
1. epoch is 10
2. mini batch size is 64
3. learning rate is 0.005

## Net Structure
#### Encode
1. Linear layer 1: transform the original picture (28 * 28) to a linear sequence (128 * 1)
2. Linear layer 2: transform from 128 * 1 to 64 * 1
3. Linear layer 3: transform from 64 * 1 to 12 * 1
4. Linear layer 4: transform from 12 * 1 to 3 * 1

#### Decode
5. Linear layer 5: transform from 3 * 1 to 12 * 1
6. Linear layer 6: transform from 12 * 1 to 64 * 1
7. Linear layer 7: transform from 64 * 1 to 128 * 1
8. Linear layer 8: transform from 128 * 1 to 28 * 28

