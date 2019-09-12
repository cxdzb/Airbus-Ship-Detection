# 卫星图像船舰识别
#### 环境要求
+ python3.7
+ 所需库：
  + numpy
  + pandas
  + matplotlib
  + pillow
  + tensorflow-gpu
  + keras-gpu
#### 项目目录：
+ data（由于训练数据过大，未全部上传，测试请使用已有模型）
+ model
  + model_build.py: 模型建立
  + model_assess.py: 模型评估
+ preprocess
  + data_deal.py: 数据处理（rle解码）
  + data_generator.py: 数据生成
  + data_split.py: 数据分割（有无船）
  + image_show.py: 数据展示
+ model_run.py: 模型训练
+ model_detect.py: 模型预测
