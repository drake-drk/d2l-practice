import numpy as np
import torch
#矩阵乘法
def logits(x,w,b):
    return x @ w.T + b

#测试
r = np.load("ref.npz")
z = logits(r["X"], r["W"], r["b"])
print("最大误差:", np.abs(z - r["logits"]).max())

#测试softmax函数
def softmax(x):
    maxv = np.max(x,axis = 1,keepdims = True)#必须是每一行减去本行的最大值，不能减去全局的最大值
    #原因：减去本行最大值，一定会至少剩下一个元素不会下溢。e**0 = 1
    xexp = np.exp(x - maxv)
    partition = xexp.sum(axis = 1,keepdims = True)
    return xexp/partition

print(softmax(np.array([[1000.0, 0.0, 0.0]])))
p = softmax(r["logits"])
print(np.allclose(p.sum(axis=1), 1))

