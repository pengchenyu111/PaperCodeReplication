import torchvision
from torch.utils.tensorboard import SummaryWriter
import ssl

# 这句话是为了解决下载数据集出现的证书失效问题
ssl._create_default_https_context = ssl._create_unverified_context

"""
    pytorch中包含的dataset的获取方式和transform转换
"""
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

print(test_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
