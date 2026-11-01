"""
我们
"""
import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

n = 10000
vect1 = torch.ones([n])
vect2 = torch.ones([n])
"""v1v2 are both vector with length=n and all elements=1"""
vect3 = torch.zeros([n])

timer = Timer() 

for i in range(n):
    vect3[i] = vect1[i] + vect2[i]
print(f'{timer.stop():.5f}second')#fstring：在{}内嵌入表达式，：是格式化指令，.5表示保留五位小数，f为定点数
"""正常计算"""

timer.start()
vect4 = vect1 + vect2
print(f'{timer.stop():.5f} sec')
"""矢量计算遥遥领先"""

"""接下来是高斯分布"""
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

x = np.arange(-7, 7, 0.01)#np.arange 的作用是生成一个等差数组。起点是 -7，终点是 7（不包含 7），步长是 0.01。
#意思是x的取值从-7到+7

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]#三组参数，分别对应期望方差（0，1）（0，2）（3，1）

plt.figure(figsize=(4.5, 2.5))#画布大小

for mu, sigma in params:
    y = normal(x, mu, sigma)
    plt.plot(x, y, label=f'mean {mu}, std {sigma}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()#显示图例
plt.show()#展示图像

