## Log for Airbus-Ship-Detection
---
### 2019.08.21
1、初步查看了数据集<br>
> * 由于数据集过于庞大和杂乱，因此亟待进行处理和分类<br>
> * 数据集标签格式为RLE，需要了解相关编码解码知识
### 2019.08.22
1、和组员讨论了之后的项目规划，完善了[第一次汇报](https://github.com/plumprc/airbus-ship-detection/tree/master/lizhe/relevent/汇报一1.2.pptx)的ppt<br>
2、实现了RLE（run-length）的解码，并成功利用掩码将标签可视化<br>
* [RLE](https://github.com/plumprc/airbus-ship-detection/tree/master/lizhe/ship-detection/RLE.ipynb) 
### 2019.08.24
1、学习了pandas表格数据处理的相关知识，实现了表格数据可视化<br>
* [pandas-groupby](https://github.com/plumprc/airbus-ship-detection/tree/master/lizhe/ship-detection/Bonus/Pandas-Groupby.ipynb)

2、针对数据集进行了预处理，随机drop了一些数据，将预选数据分为training set和validation set
* [split](https://github.com/plumprc/airbus-ship-detection/tree/master/lizhe/ship-detection/split.ipynb)
### 2019.08.25
1、学习了第一个卷积神经网络LeNet的相关知识<br>
2、用Pytorch实现了简单的LeNet-5并完成了可视化工作
* [LeNet-5](https://github.com/plumprc/airbus-ship-detection/tree/master/lizhe/ship-detection/Bonus/LeNet-5.ipynb)
> * 亟待学习卷积核的工作机理并给出一般卷积核的选择规律
> * 继续巩固神经网络的相关知识及搭建工作，实现一个最基本的ResNet-34模型并了解相关机理