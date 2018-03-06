# Sigmoid激活函数类
from functools import reduce
from time import sleep

import numpy as np
from FullConnectedLayer import FullConnectedLayer


class SigmoidActivator(object):
    def forward(self, weighted_input):
        # 用于预测
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        # 对sigmoid函数求导，用来求出梯度，用于反向传播，
        return output * (1 - output)

# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        # 此处为大坑，因为sample是个一维数组，这里必须将output变为二维数组,否则最后算出来的shape有问题
        output = np.array(output).reshape(784,1)
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                    data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        # 先对结果进行预测
        self.predict(sample)
        # 然后根据预测结果计算误差项，得到梯度
        self.calc_gradient(label)
        # 最后根据梯度进行权值更新
        self.update_weight(rate)

    def calc_gradient(self, label):
        # 此处也是个大坑，label是个一维数组，必须转换为二维数组，否则label - self.layers[-1].output的shape会有问题
        label = np.array(label).reshape(10,1)
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

'''
    def gradient_check(self,sample,label):
        # 计算网络误差
        network_error = lambda vec1, vec2: 0.5* reduce (lambda a, b: a + b,
                         map(lambda v1,v2: (v1 - v2) * (v1 - v2),
                             vec1,vec2))
        self.predict(sample)
        self.calc_gradient(label)
        epsilon = 10e-4
        actual_error = network_error(self.predict(sample),label)
        for layer in self.layers:
            layer.W += epsilon
        error1 = network_error(self.predict(sample),label)
        for layer in self.layers:
            layer.W += 2*epsilon
        error2 = network_error(self.predict(sample), label)
        expected_error = (error2 - error1) / (2 * epsilon)
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_error, actual_error))
'''



