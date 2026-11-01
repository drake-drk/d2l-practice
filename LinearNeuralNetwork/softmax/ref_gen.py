import torch, torch.nn as nn, numpy as np

torch.manual_seed(0)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
#全连接层，784个输入，10个输出
net[1].weight.data.normal_(0, 0.01)#net[1]就是线性层
net[1].bias.data.zero_()

X = torch.randn(32, 1, 28, 28)#randn：生成符合标准正态分布的随机数。(32, 1, 28, 28)：生成 32 张图片。每张图片是 1 个通道（黑白），高 28 像素，宽 28 像素。
y = torch.randint(0, 10, (32,))#randint(0, 10, ...)：在 0 到 9 之间随机抽数字。(32,)：生成一个长度为 32 的列表。代表这 32 张图分别属于哪一类。
#意思是：x是32个照片的所有像素的值。y是每个照片的对应标签


out  = net(X)#对32张图的每一张都给出了10个评分，所以形状为32*10
loss = nn.CrossEntropyLoss()(out, y)#交叉熵损失函数
loss.backward()#反向传播

np.savez("ref.npz",
         X=X.numpy().reshape(32, -1), y=y.numpy(),
         W=net[1].weight.detach().numpy(),
         b=net[1].bias.detach().numpy(),
         logits=out.detach().numpy(),
         loss=np.array(loss.item()),
         gW=net[1].weight.grad.numpy(),
         gb=net[1].bias.grad.numpy())
print("W", net[1].weight.shape, "logits", out.shape, "loss", loss.item())