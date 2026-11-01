import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import IPython.display as display

# 1. 数据加载
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()

# 加载FashionMNIST训练集和测试集
mnist_train = torchvision.datasets.FashionMNIST(#训练集
    root="../data", train=True, transform=trans, download=True)

mnist_test = torchvision.datasets.FashionMNIST(#测试集
    root="../data", train=False, transform=trans, download=True)

print(f'训练集大小: {len(mnist_train)}, 测试集大小: {len(mnist_test)}')

# 数据加载器
batch_size = 256
train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

# 2. 定义模型
# 定义一个简单的Softmax回归模型，使用PyTorch的nn.Sequential
num_inputs = 784  # 28x28像素
num_outputs = 10  # 10个类别

net = nn.Sequential(
    nn.Flatten(),  # 将28*28图像展平为一维向量784*`1
    nn.Linear(num_inputs, num_outputs)  # 全连接层
)

# 初始化模型参数
net[1].weight.data.normal_(0, 0.01)
net[1].bias.data.zero_()

# 3. 损失函数
# 使用PyTorch内置的CrossEntropyLoss（已经包含了softmax计算）
criterion = nn.CrossEntropyLoss()

# 4. 优化器
# 使用随机梯度下降优化器
lr = 0.1
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 5. 辅助函数
# 获取Fashion-MNIST数据集的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 计算模型精度
def evaluate_accuracy(net, data_iter):
    net.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    net.train()  # 将模型设置回训练模式
    return correct / total

# 训练动画类，用于可视化训练过程
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, figsize=(5, 3)):
        if legend is None:
            legend = []
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        self.fig, self.axes = plt.subplots(figsize=figsize)
        self.config_axes = lambda: self.set_axes(self.axes, xlabel, ylabel, xlim, ylim, legend)
        self.X, self.Y, self.fmts = None, None, ['-', 'm--', 'g-.', 'r:']

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes.cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 6. 训练模型
def train(net, train_iter, test_iter, criterion, optimizer, num_epochs=10):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        total_train = 0
        for X, y in train_iter:
            # 前向传播
            outputs = net(X)
            loss = criterion(outputs, y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失和精度
            train_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == y).sum().item()
            total_train += X.size(0)
        
        # 计算训练集和测试集的平均损失和精度
        avg_train_loss = train_loss / total_train
        avg_train_acc = train_acc / total_train
        test_acc = evaluate_accuracy(net, test_iter)
        
        # 更新动画
        animator.add(epoch + 1, (avg_train_loss, avg_train_acc, test_acc))
    print(f'训练完成！最终测试精度: {test_acc:.4f}')

# 7. 运行训练
num_epochs = 10
train(net, train_iter, test_iter, criterion, optimizer, num_epochs)

# 8. 模型预测和可视化
def show_predictions(net, test_iter, n=6):
    for X, y in test_iter:
        break
    with torch.no_grad():
        outputs = net(X)
        predicted = torch.argmax(outputs, dim=1)
    
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(predicted)
    
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    for i in range(n):
        img = X[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'真: {trues[i]}\n 预: {preds[i]}')
        axes[i].axis('off')
    plt.show()

# 显示预测结果
show_predictions(net, test_iter)
