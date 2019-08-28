#@Time   :2019/8/27 19:49
#@author : qtgavc

import torchvision

# model = torchvision.models.resnet18(pretrained=False)
# print(model)

# model = torchvision.models.resnet34(pretrained=False)
# print(model)

model = torchvision.models.resnet50(pretrained=False)
print(model)

model = torchvision.models.resnet152(pretrained=False)
print(model)