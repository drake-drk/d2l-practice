import time

import matplotlib.pyplot as plt  # 替代 d2l.plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# 1. 图像数据变换
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()


# 2. 辅助函数：获取文本标签
def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 3. 辅助函数：绘制图像列表 (替换 d2l.plt)
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    # 改动点：d2l.plt.subplots -> plt.subplots
    # 返回一个画板和一堆格子，只要格子，就用 _ 表示不需要画板
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # 假设原本6行7列，在此拉长为1*42
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


# 4. 封装的数据加载函数
def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def main():
    batch_size = 256

    # 1. 图像数据加载
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    print(f'训练集大小: {len(mnist_train)}, 测试集大小: {len(mnist_test)}')

    # 2. 测试可视化：取 18 张图展示
    # 一次取18张图的数据和对应的标签
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # 把数据拼成图片，18张图，像素为28*28，用标签找到对应英文名称
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    plt.show()  # 原生 matplotlib 需要 show() 才能显示图像

    # 3. 测试数据加载速度
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers())
    start_time = time.perf_counter()
    for X, y in train_iter:
        continue
    elapsed_time = time.perf_counter() - start_time
    print(f'{elapsed_time:.2f} sec')

    # 4. 测试 resize 功能
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break


if __name__ == '__main__':
    # Windows / Python 3.14(默认 forkserver) 下 num_workers>0 必须有这层保护：
    # 否则子进程会重新执行整个脚本，导致 ConnectionResetError / RuntimeError。
    main()
