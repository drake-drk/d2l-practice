import random
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def synthetic_data(w, b, num_examples):  
    #生成y=Xw+b+噪声
    X = torch.normal(0, 1, (num_examples, len(w)))#在正太分布中挑选一个数字
    #num_examples = 1000, len(w) = 2.意思是从正太分布中挑选1000个数字，每个数字有2个特征
    #也就是说，x是1000 * 2的矩阵，每个元素都服从标准正太分布

    y = torch.matmul(X, w) + b
    #w是2 * 1的向量，因为两个输入对应着两个权重。b是一个标量    
    #matmul()负责进行矩阵乘法，1000 * 2 * 2 * 1 = 1000 * 1
    #所以y是1000*1的向量
    #注意，此时的y无噪音

    y += torch.normal(0, 0.01, y.shape)#为y添加噪音，噪音服从（0，0.01）的正态分布。y.shape可以确保y的每个元素都能加上噪音
    return X, y.reshape((-1, 1))#y.reshape()把y变成1000个样本，每个样本是一个标量，即1000*1的矩阵，每个元素都是一个标量
    #[x1,x2...x1000]->[[x1],[x2]...[x1000]].统一格式方便训练模型


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
#features即是特征，对应返回的x，是1000*2的矩阵，labels对应y.shape，1000*1的向量
print('features:', features[0],'\nlabel:', labels[0])#打印1000个样本中第一个样本的特征和标签

#因为不需要求梯度，使用detach，防止记录不必要的计算图影响性能
features_np = features.detach().numpy()
labels_np = labels.detach().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(features_np[:, 1], labels_np, 1)
#scatter创建散点图，features[:,1]即是x的所有行的第二列，第二个特征
#综上所述，这张图反映的是y与特征2之间的关系
plt.show()

#当然我们也可以用三维散点图同时反映y和两个特征的关系
import matplotlib.pyplot as plt
# 创建一个新的图形
fig = plt.figure(figsize=(8, 6))
# 添加一个 3D 子图
ax = fig.add_subplot(111, projection='3d')
# 绘制三维散点图
scatter = ax.scatter(features_np[:, 0], features_np[:, 1], labels_np)
# 设置坐标轴标签
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Label (y)')
# 显示图像
plt.show()

# 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。
# 每个小批量包含一组特征和标签。
#把大数据集分成一批批小数据集并且不断迭代循环更新模型
def data_iter(batch_size, features, labels):
    num_examples = len(features)#1000
    indices = list(range(num_examples))#生成值为0到999的，大小为1000的数组
    random.shuffle(indices)#打乱数组内的数字顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])#这一行意思是,取indices数组中从i到i+batch_size的元素作为一个1*batch_size的张量并命名为batch_indices
        yield features[batch_indices], labels[batch_indices]
        #yield不同于return，yield返回的是一个迭代器，而return返回的是一个值
        #在每次迭代中，yield返回一个小批量的特征和标签，而不是一次返回所有数据
        #这使得数据迭代更加高效，因为它不需要在内存中存储所有数据

batch_size = 10#每个组内有十个样本

""""
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break#打印完第一批后就中止
"""

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
#w是2*1的向量，因为有两个特征，每个特征对应一个权重，两个都服从（0，0.01）的正态分布
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
    #x与w作矩阵乘法加上偏置值

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    #y_hat是预测值，y是真值

def sgd(params, lr, batch_size):  #@save stochastic gradient descent
    #params是模型参数，lr学习率用于控制步长
    """小批量随机梯度下降"""
    with torch.no_grad():#接下来的操作不用记录梯度
        for param in params:
            param -= lr * param.grad / batch_size
            #除以batchsize是为了得到平均梯度
            param.grad.zero_()#清空梯度

lr = 0.03
num_epochs = 3#循环轮数，每一轮都会遍历所有样本
net = linreg#线性回归模型
loss = squared_loss#损失函数用于计算均方损失

for epoch in range(num_epochs):
    #一共三个epoch，每轮都要计算1000个。
    for X, y in data_iter(batch_size, features, labels):#1000个会十个十个（batchsize）地计算
        l = loss(net(X, w, b), y)#net（）调用模型，基于小样本计算出y_hat
        #loss()利用y_hat和y计算均方损失
        l.sum().backward()#对所有样本的损失求和再用backward（）计算梯度
        sgd([w, b], lr, batch_size)#再用sgd更新模型
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)#计算当前最优模型的loss
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        #打印当前轮数和它的平均损失

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

print(f"w: {w.detach().numpy()}")#打印当前最优模型的w
print(f"b: {b.detach().numpy()}")#打印当前最优模型的b