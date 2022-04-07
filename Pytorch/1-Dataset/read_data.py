from torch.utils.data import Dataset
import os
from PIL import Image


class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir):
        """
        初始化函数，
        self.属性表示将此属性变为全局属性，方便其他方法调用
        :param root_dir: 根路径
        :param image_dir: 图片路径
        :param label_dir: 标签路径
        """
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        """
        获取每个item
        :param idx: 列表下标
        :return:
        """
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        # 获取图片
        img = Image.open(img_item_path)
        # 获取图片label
        with open(label_item_path, 'r') as f:
            label = f.readline()
        return img, label

    def __len__(self):
        """
        获取整个数据集的长度
        :return:
        """
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


if __name__ == '__main__':
    root_dir = "../dataset/hymenoptera_data/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees)
    # 可以对两个数据集直接进行拼接
    train_dataset = ants_dataset + bees_dataset

    print(ants_dataset[0])
