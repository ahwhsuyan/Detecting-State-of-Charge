import torch
import torch.nn as nn
import math
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import einops
from torch import nn
from torch.functional import norm
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.nn import Softmax

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)
    
class EfficientAdditiveAttention(nn.Module):
    """
    高效加性注意力模块，用于SwiftFormer中。
    输入：形状为[B, N, D]的张量
    输出：形状为[B, N, D]的张量
    """

    def __init__(self, in_dims=512, token_dim=512):
        super().__init__()
        # 初始化查询和键的线性变换
        self.to_query = nn.Linear(in_dims, token_dim)
        self.to_key = nn.Linear(in_dims, token_dim)

        # 初始化可学习的权重向量和缩放因子
        self.w_a = nn.Parameter(torch.randn(token_dim, 1))
        self.scale_factor = token_dim ** -0.5

        # 初始化后续的线性变换
        self.Proj = nn.Linear(token_dim, token_dim)
        self.final = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        x = x.view(1,1072,2)
        B, N, D = x.shape  # B:批次大小，N:序列长度，D:特征维度

        # 生成初步的查询和键矩阵
        query = self.to_query(x)
        key = self.to_key(x)

        # 对查询和键进行标准化处理
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # 学习查询的注意力权重，并进行缩放和标准化
        query_weight = query @ self.w_a
        A = query_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        # 通过注意力权重对查询进行加权，以生成全局查询向量
        q = torch.sum(A * query, dim=1)
        q = q.reshape(B, 1, -1)

        # 计算全局查询向量和每个键的交互，再与原始查询进行逐元素相加
        out = self.Proj(q * key) + query
        out = self.final(out)  # 通过最终的线性层输出调制后的特征
        out = out.reshape(1072, 2)
        return out
# 定义ECA注意力模块的类
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        
        # self.conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)#zengjiade
        
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        
        # y = self.conv2d(y)  # 新增的2D卷积层，用于进一步处理特征，保持维度为bs,c,1,1######新加的
        
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法

class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准

class DetectionModelDNN(nn.Module):
   def __init__(self, hidden_size, input_size, p):
    super(DetectionModelDNN, self).__init__()
    self.Ef = EfficientAdditiveAttention(in_dims=2, token_dim=2)
    self.ECA=ECAAttention()
    self.simam = Simam_module()
    
    self.network = nn.Sequential(      
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )

   def forward(self, input):

    input = self.network(input)  
    
    #Feature Aggregation stage
    input = input.view(1, 1, 1072, 2)  #  (b, c, h, w)
    input1 = self.simam(input)
    input2 = self.ECA(input)
    input = input1+input2   
    input = input.view(1, 1072, 2)  #  (b, (h w), c) 
    input = self.Ef(input)
    
    #Feature Repeat Refining phase
    input = input.view(1, 1, 1072, 2)  #  (b, c, h, w)
    input = self.simam(input)
    input = self.ECA(input)
    input = input.view(1, 1072, 2)  #  (b, (h w), c) 
    input = self.Ef(input)
    
    #Final Feature Processing stage
    input = input.view(1, 1, 1072, 2)  #  (b, c, h, w)
    input = self.simam(input)
    input = input.view(1, 1072, 2)  #  (b, (h w), c) 
    input = self.Ef(input)
    
    input.view(1072,2)
    
    return input

