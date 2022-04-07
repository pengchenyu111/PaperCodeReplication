from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

"""
    TensorBoard主要用来对训练过程中的参数等数据做可视化，比如你可以看到训练过程中loss、梯度等数据的变化。
    1、使用之前先安装TensorBoard包：
        conda install TensorBoard
    2、编写代码，展示需要可视化的数据：
    
    3、使用命令启动TensorBoard页面;
        tensorboard --logdir=Pytorch/2-TensorBoard/logs --port=6007
"""

# SummaryWriter中的核心参数为事件文件保存位置
writer = SummaryWriter("logs")

image_path = "../dataset/hymenoptera_data/train/ants_image/67270775_e9fdf77e9d.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

# 注意add_image的img_tensor只能是torch.Tensor, numpy.array, or string/blobname类型，而且还要注意图片的格式，详情Ctrl
writer.add_image(tag="train", img_tensor=img_array, global_step=1, dataformats='HWC')

writer.close()
