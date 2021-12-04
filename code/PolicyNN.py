# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from collections import deque

import tensorflow.keras.backend as K
import random
import numpy as np
import pickle

from tqdm import tqdm
from tkinter import *

# 用以搭建通用得到残差网络结构
class ResidualCNN():
    L2 = 1e-4  # L2正则化系数
    def __init__(self, input_dim, output_dim, hidden_layers = {"filters":38, "kernel_size":(3, 3)}, num_layers = 5):
        self.input_dim = input_dim              # 模型输入尺寸
        self.output_dim = output_dim            # 模型输出尺寸
        self.hidden_layers = hidden_layers      # 隐藏层层数
        self.num_layers = num_layers            # 残差网络层数
        self.filters = hidden_layers["filters"]
        self.kernel_size = hidden_layers["kernel_size"]
        self.model = self.BuildModel()  # 搭建模型
    
    # 卷积层模板
    def ConvLayer(self, input_block, filters, kernel_size):
        output = Conv2D(
            filters = filters,              # 卷积滤波器个数
            kernel_size = kernel_size,      # 卷积滤波器的尺寸 (3,3),(5,5)
            data_format="channels_first",   # 表示输入张量中维度的顺序
            padding = 'same',               # 边缘填充
            use_bias=False,                 # 该层是否使用偏置向量
            activation='linear',            # 线性激活函数
            kernel_regularizer = l2(self.L2) # L2正则化项
        )(input_block)
        output = BatchNormalization(axis=1)(output) # 批量标准化层 
        output = LeakyReLU()(output)
        return (output)
    
    # 残差卷积层模板
    def ResLayer(self, input_block, filters, kernel_size):
        output = Conv2D(
            filters = filters,              # 卷积滤波器个数
            kernel_size = kernel_size,      # 卷积滤波器的尺寸 (3,3),(5,5)
            data_format="channels_first",   # 表示输入张量中维度的顺序
            padding = 'same',               # 边缘填充
            use_bias=False,                 # 该层是否使用偏置向量
            activation='linear',            # 线性激活函数
            kernel_regularizer = l2(self.L2) # L2正则化项
        )(input_block)
        output = BatchNormalization(axis=1)(output) # 批量标准化层 
        output = LeakyReLU()(output)                # 激活函数

        output = Conv2D(
            filters = filters,              # 卷积滤波器个数
            kernel_size = kernel_size,      # 卷积滤波器的尺寸 (3,3),(5,5)
            data_format="channels_first",   # 表示输入张量中维度的顺序
            padding = 'same',               # 边缘填充
            use_bias=False,                 # 该层是否使用偏置向量
            activation='linear',            # 线性激活函数
            kernel_regularizer = l2(self.L2) # L2正则化项
        )(output)
        output = BatchNormalization(axis=1)(output) # 批量标准化层 
        output = add([input_block, output])         # 拼接输入各个张量的和
        output = LeakyReLU()(output)                # 激活函数
        return (output)

    def ValueHead(self, output):
        conv = Conv2D(
            filters = 1 ,                  
            kernel_size = (1,1) , 
            data_format="channels_first" , 
            padding = 'same', 
            use_bias=False, 
            activation='linear', 
            kernel_regularizer = l2(self.L2)
        )(output)
        conv = BatchNormalization(axis=1)(conv)
        conv = LeakyReLU()(conv)
        flatten = Flatten()(conv)

        dense = Dense(
            32, # 输出空间维度
            use_bias=False, 
            activation='linear', 
            kernel_regularizer=l2(self.L2)
        )(flatten)
        dense = LeakyReLU()(dense)
        dense = Dense(
            1, 
            use_bias=False, 
            activation='tanh', 
            kernel_regularizer=l2(self.L2), 
            name = 'ValueHead'
        )(dense)
        return (dense)

    def PolicyHead(self, output):
        conv = Conv2D(
            filters = 2, 
            kernel_size = (1,1), 
            data_format="channels_first", 
            padding = 'same', 
            use_bias=False, 
            activation='linear', 
            kernel_regularizer = l2(self.L2)
        )(output)
        conv = BatchNormalization(axis=1)(conv)
        conv = LeakyReLU()(conv)
        conv = Flatten()(conv)
        dense = Dense(
            self.output_dim, 
            use_bias=False, 
            activation='softmax', 
            kernel_regularizer=l2(self.L2), 
            name = 'PolicyHead'
        )(conv)
        return (dense)

    # 搭建模型
    def BuildModel(self):
        input = Input(self.input_dim, name = 'Input')
        conv = self.ConvLayer(input, self.filters, self.kernel_size)
        
        # 中间就是若干的残差网络
        for _ in range (self.num_layers):
            conv = self.ResLayer(conv, self.filters, self.kernel_size)

        # 落子决策的概率分布
        self.policyNet = self.PolicyHead(conv)

        # 预测该落子之后，赢（输）的概率
        self.valueNet = self.ValueHead(conv)

        model = Model(inputs = [input], outputs = [self.policyNet, self.valueNet])

        model.compile(
            loss={
                'ValueHead': 'mean_squared_error', 
                'PolicyHead': 'categorical_crossentropy'
            },
			optimizer=Adam(),	
			loss_weights={
                'ValueHead': 0.5, 
                'PolicyHead': 0.5
            })
        return model

class PolicyValueNet(ResidualCNN):
    """
    当面对每一个棋盘状态时，预测出每一个位置的落子概率，以及该位置的胜率
    """
    trainDataPoolSize = 18000*2 # 用以存储训练网络的数据
    trainBatchSize = 1024*2     # 每次从数据池(trainDataPool)中随机采样出的一批训练数据
    epochs = 10                 # 每次训练步数
    
    trainDataPool = deque(maxlen=trainDataPoolSize) # 训练数据池
    
    kl_targ = 0.02
    learningRate = 2e-3 # 学习率 
    LRfctor = 1.0       # 适应性地调整学习率

    def __init__(self, input_dim):
        self.input_dim = input_dim
        ResidualCNN.__init__(self, input_dim = self.input_dim, output_dim = self.input_dim[1]*self.input_dim[2])
        
    def policy_NN(self, input):
        """
        input: 输入的棋盘状态
        output: 每个可以落子的（行动，胜率）tuples列表，以及棋盘状态回报。
        """
        emptySpace = input.availables        # 棋盘目前的空位(即可以落子的地方)
        currentBoard = input.current_state() # 获取当前棋盘状态

        currentBoard = currentBoard.reshape(-1, self.input_dim[0], self.input_dim[1], self.input_dim[1])
        currentBoard = np.array(currentBoard)

        # 根据当前的棋盘作为模型的输入，预测出当前的(落子+胜率)预测
        act_probs, value = self.model.predict_on_batch(currentBoard)
        act_probs = zip(emptySpace, act_probs.flatten()[emptySpace])

        return act_probs, value[0][0]

    def train(self, batchBoard, batchProbs, batchWinner, learning_rate):
        # 设置数据的格式
        oneBatchBoard  = np.array(batchBoard)
        oneBatchProbs  = np.array(batchProbs)
        oneBatchWinner = np.array(batchWinner)
        
        # 评价当前模型
        loss = self.model.evaluate(oneBatchBoard, [oneBatchProbs, oneBatchWinner], batch_size=len(batchBoard), verbose=0)
        # 重新设置模型的学习率
        K.set_value(self.model.optimizer.lr, learning_rate)
        # 训练当前模型
        self.model.fit(oneBatchBoard, [oneBatchProbs, oneBatchWinner], batch_size=len(batchBoard), verbose=0)
        return loss[0]
    
    def update(self, scrollText):
        """
        更新预测网络的参数
        """
        # 从数据池中随即采样一批数据进行训练
        trainBatchSize = random.sample(self.trainDataPool, self.trainBatchSize)
        
        # trainBatchSize 三元组
        # 棋盘状态 + 落子概率分布 + 胜者预测
        batchBoard  = [data[0] for data in trainBatchSize]
        batchProbs  = [data[1] for data in trainBatchSize]
        batchWinner = [data[2] for data in trainBatchSize]

        # 返回这一批训练数据样本的模型预测值
        # 也就是首先记录下当前模型的预测水平
        batchProbsOld, batchValueOld = self.model.predict_on_batch(np.array(batchBoard))

        pbar = tqdm(range(self.epochs),ncols=38)
        for epoch in pbar:
            scrollText.delete(1.0, END)
            scrollText.insert(END, '正在训练NN\n'+str(pbar)+'\n')
            scrollText.see(END)
            scrollText.update()

            # 根据这批训练数据，对模型进行训练
            loss = self.train(batchBoard, batchProbs, batchWinner, self.learningRate*self.LRfctor)

            # 返回训练之后模型预测值
            batchProbsNew, batchValueNew = self.model.predict_on_batch(np.array(batchBoard))

            # 计算Kullback-Leibler散度 恒两个前后两次 概率分布预测值的差异
            kl = np.mean(np.sum(batchProbsOld * (np.log(batchProbsOld + 1e-10) - np.log(batchProbsNew + 1e-10)),axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度严重发散，则提前停止使用
                break

        scrollText.delete(1.0, END)
        scrollText.insert(END, '正在训练NN\n'+str(pbar)+'\n训练结束 loss: '+str(round(loss,4))+'\n')
        scrollText.see(END)
        scrollText.update()


        # 更新学习率
        if kl > self.kl_targ * 2 and self.LRfctor > 0.1:
            self.LRfctor /= 1.5
        elif kl < self.kl_targ / 2 and self.LRfctor < 10:
            self.LRfctor *= 1.5
        
        # print(("kl:{:.5f}," "LRfctor:{:.3f}," "loss:{}," ).format(kl, self.LRfctor, loss))
        return loss

    def memory(self, play_data):
        """
        存储训练数据
        play_data: [(棋盘状态, 落子概率, 胜者预测), ..., ...]
        """
        # 将收集到的自我对弈数据进行《数据增强》
        play_data = self.get_DataAugmentation(list(play_data)[:])
        self.trainDataPool.extend(play_data) # 加入训练数据池中


    def load_model(self, model_file):
        """
        加载模型
        model_file: 模型存储路径
        """ 
        NNParams = pickle.load(open(model_file, 'rb')) # 读取模型
        self.model.set_weights(NNParams)               # 设置模型参数
        # print("正在加载模型: " + model_file)


    def save_model(self, model_file):
        """ 
        保存模型
        model_file: 模型存储路径
        """
        NNParams = self.model.get_weights() # 获取模型参数
        pickle.dump(NNParams, open(model_file, 'wb'), protocol=2) # 保存
        # print("保存模型到文件: " + model_file)


    def get_DataAugmentation(self, play_data):
        """
        通过旋转和翻转来增加数据集
        play_data: [(棋盘状态, 落子概率, 胜者预测), ..., ...]
        """
        # 扩展之后的数据集
        extendData = []
        for board, porbs, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                # -------------------------------------------------
                # 旋转每一个棋盘棋子状态
                equi_board = np.array([np.rot90(b,i) for b in board])
                # 旋转每一个棋盘上的概率分布
                equi_porbs = np.rot90(np.flipud(porbs.reshape(self.input_dim[1], self.input_dim[2])), i)
                extendData.append((equi_board, np.flipud(equi_porbs).flatten(), winner)) # 扩展数据
                
                # 水平翻转
                # -------------------------------------------------
                equi_board = np.array([np.fliplr(s) for s in equi_board])
                equi_porbs = np.fliplr(equi_porbs)
                extendData.append((equi_board, np.flipud(equi_porbs).flatten(), winner)) # 扩展数据

        return extendData