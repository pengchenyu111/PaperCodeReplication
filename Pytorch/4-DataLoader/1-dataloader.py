import torchvision
import ssl

# 这句话是为了解决下载数据集出现的证书失效问题
ssl._create_default_https_context = ssl._create_unverified_context

"""
    有时候一个数据集太大
    DataLoader可以很好的控制怎么从原始数据集中抓取数据
    参数含义：
        dataset：原始数据集
        batch_size：每次抓取的数据集大小
        shuffle：是否打乱顺序抓取
        num_workers: 多进程抓取
        drop_last: 按batch_size抓后剩下的数据是否丢弃
"""
# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
