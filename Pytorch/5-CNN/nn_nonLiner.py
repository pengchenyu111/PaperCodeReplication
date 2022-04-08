import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


tudui = Tudui()

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input-nonLiner", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output-nonLiner", output, step)
    step += 1

writer.close()
