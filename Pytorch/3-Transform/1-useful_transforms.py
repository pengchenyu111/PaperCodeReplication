from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
    Transform的功能就是：转换数据格式
"""

writer = SummaryWriter("logs")
img = Image.open("../dataset/hymenoptera_data/train/ants_image/28847243_e79fe052cd.jpg")
print(img)

# ToTensor
# 将数据转换为tensor的格式，tensor格式是一种机器学习对象的数据格式，里面包含了如梯度、传播等参数
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor)
writer.add_image("ToTensor", img_tensor)

# Compose()可以将多个转换操作组合在一起
trans_com = transforms.Compose([transforms.Resize((250, 250)), transforms.ToTensor()])
img_com = trans_com(img)
print(img_com)
writer.add_image("Compose", img_com)


writer.close()