import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


# 打印时间
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


# pc_normalize为point cloud normalize
# 即将点云数据进行归一化处理
# 归一化点云，使用已centroid为中心的坐标，球半径为1
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)  # 压缩点云数据求得x,y,z的均值
    pc = pc - centroid  # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 求得离中心点最大距离，最大的标准差
    pc = pc / m  # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
    return pc


# 确定每个点到采样点的距离，用于ball_query过程
# 欧式距离
# 函数输入是两组点，N为第一组点src个数，M为第二组点dst个数，C为输入点的通道数（如果xyz时C=3）
# 函数返回的是两组点两两之间的欧式距离，即N*M矩阵
# 函数训练中数据以Mini-Batch的形式输入，所以一个Batch数量的维度为B
def square_distance(src, dst):
    # 由于在训练中数据通常是以Mini-Batch的形式输入的
    # 所以有一个Batch数量的维度为B。
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # matmul矩阵相乘，2*(xn * xm + yn * ym + zn * zm)
    # 为了保证src和dst矩阵可以相乘，这里涉及到三维矩阵乘法
    # 需要将dst转变一下维度[B, C, M]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # xn*xn + yn*yn + zn*zn
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # xm*xm + ym*ym + zm*zm
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# 按照输入的点云数据和索引返回索引的点云数据
# 例如Points为B*2048*3点云，idx为[5.666，1000.2000]
# 则返回Batch中第5666，1000，2000个点组成的B*4*3的点云集
# 如果idx为一个[B,D1,''''DN],则它会按照idx中的纬度结构将其提取成[B，D1，‘’‘DN,C]
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]#   #点云数据points
        idx: sample index data, [B, S]   #点云索引idx
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # idx为每各样本中所要选取的点的索引
    # 这里输入的点云数据B*2048*3，其中B为Batch_size，样本数
    # 简单来说，这个函数就是要再点云数据中选取每个样本中索引值在idx这个索引数组里面的点
    # idx的长度为4时，则最后输出的为B*4*3，也就是在2048个点中选取在索引值idx的点
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape=[B,1]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # repeat_shape=[B,S]
    repeat_shape = list(idx.shape)
    # repeat_shape=[B,S]
    repeat_shape[0] = 1
    # .view(view_shape)=.view(B,1)
    # .repeat(repeat_shape)=.view(1,S)
    # 综上所述，batch_indices的维度[B,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # 从points当中取出每个batch_indices对应索引的数据点
    new_points = points[batch_indices, idx, :]
    return new_points


# 最远点采样
# farthest_point_sample函数完成最远点采样
# 从一个输入点云中按照所需要的点的个数npoint采样出足够多的点
# 并且点与点的距离需要足够远
# 返回结果是npoint个采样点再原始点云中的索引
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape  # B：Batch_size, N:num_points, C:channel
    # 初始化一个centrdis矩阵，用于存储npoint个采样点的索引位置，大小为B*npoint
    # 其中B为BatchSize的个数
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 8*512
    # distance矩阵（B*N）用来记录batch中所有点到某一个点的距离，初始化值很大，后面会迭代更新
    distance = torch.ones(B, N).to(device) * 1e10  # 8*1024
    # farthest表示当前最远的点,也是随机初始化,范围为0-N,初始化B个;每个batch都随机有一个初始化最远点  # 记录某个样本中所有点到某一个点的距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # batch里每个样本随机初始化一个最远点的索引
    # batch_indices初始化为0-(B-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):  # 更新第i个最远点
        # 假设当前采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest  # 第一个采样点选随机初始化的索引
        # 取出该中心点centroid的坐标 # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 得到当前采样点的坐标 B*3  # 计算点集中的所有点到这个最远点的欧式距离
        # 取出该中心点centroid点的欧式距离,存到dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算当前采样点与其他点的距离    # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        # 建立一个mask,如果dist中的元素小于distance矩阵中保存的距离值,则更新distance中的对应值
        # 随着迭代的继续,distance矩阵中的值会慢慢变小
        # 其相当于记录着某个batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance  # 选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]  # 从distance矩阵中去除最远的点为farthest,继续下一轮迭代
        farthest = torch.max(distance, -1)[1]  # 重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids


# query_ball_point函数用于寻找球形邻域中的点
# 输入中radius为球形邻域的半径，nsample为每个邻域中要采样的点
# new_xyz为centroids点的数据，xyz为所有的点云数据
# 输出为每个样本的每个球形邻域的nsample个采样点集的索引【B，S,nsample】
# 寻找球形领域中的点
def query_ball_point(radius, nsample, xyz, new_xyz):
    # 输入中radius为球形领域的半径
    # nsample为每个领域中要采样的点
    # new_xyz为S个球形领域的中心（由最远点采样在前面得出）
    # xyz为所有的点云
    # 输出为每个样本的每个球形领域的nsample个采样点集的索引
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists:[B,S,N]记录S个中心点（new_xyz）与所有点（xyz）之间的欧氏距离
    sqrdists = square_distance(new_xyz,
                               xyz)  # 得到B N M （就是N个点中每一个和M中每一个的欧氏距离）   # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    group_idx[
        sqrdists > radius ** 2] = N  # 找到距离大于给定半径的设置成一个N值（1024）索引 # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :,
                :nsample]  # 做升序排序，后面的都是大的值（1024）# 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # 如果半径内的点没那么多，就直接用第一个点来代替了。。。
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


# Sampling+Grouping主要用于将整个点云分散成局部的group
# 对于每一个group都可以用PointNet单独的提取局部的全局特征
# Sampling+Grouping分成了sampl_and_group和sampl_and_group_all两个函数
# 其区别在于sample_and_group_all直接将所有点作为一个group
# 例如：
# 512=npoint:poins sampled in farthest point sampling
# 0.2=radius:search radius in local region
# 32=nsample:how many points in each local region
# 将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 从原点云通过最远点采样挑出的采样点作为new_xyz：
    # 先用farhest_point_sample函数实现最远点采样得到采样点的索引
    # 再通过index_points将这些点的从原始点中挑出来，作为new_xyz
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    torch.cuda.empty_cache()
    # 从原点云中挑出最远点采样的采样点为new_xyz
    new_xyz = index_points(xyz, fps_idx)  # 中心点# new_xyz代表中心点，此时维度为[B, S, 3]
    torch.cuda.empty_cache()
    # idx:[B,npoint,nsample]，代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()  # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    # grouped_xyz:[B, npoint, nsample, C]
    # 通过index_points将所有group内的nsample个采样点从原始点中挑出来
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    # grouped_xyz减去采样点即中心值
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()
    # 如果每个点上有新的特征维度，则拼接新的特征与旧的特征，否则直接返回旧的特征
    # 注：用于拼接点的特征数据和点坐标数据
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        # 通过index_points将所有group内的nsample个采样点从原始点中挑出来，得到group内点的除坐标维度外的其他维度的数据
        grouped_points = index_points(points, idx)
        # dim=-1代表按照最后的维度进行拼接，即相当于dim=3
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


# sample_and_group_all直接将所有点作为一个group；n_point=1
# 将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征
def sample_and_group_all(xyz, points):
    # 与前面的不同在于：直接将所有点作为一个group，即增加一个长度为1的维度而已
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # new_xyz代表中心点，用原点表示
    new_xyz = torch.zeros(B, 1, C).to(device)
    # grouped_xyz减去中心点：每个区域的点减去区域的中心值，由于中心点为原点，所以结果仍然是grouped_xyz
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        # view(B, 1, N, -1)，-1代表自动计算，即结果等于view(B, 1, N, D)
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# PointNetSetAbstraction类实现普通的SetAbstraciton：
# 然后通过sample_and_group的操作形成局部group
# 然后对局部group中的每一个点做MLP操作，最后进行局部最大池化，得到局部的全局特征
class PointNetSetAbstraction(nn.Module):
    # 例如：npoint=128,radius=0.4,nsample=64,in_channle=128+3,mlp=[128,128,256],group_all=False
    # 128=npoint:points sampled in farthest point sampling
    # 0.4=radius:search radius in local region
    # 64=nsample:how many points inn each local region
    # [128,128,256]=output size for MLP on each point
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        # 在构造函数__init__中用到list、tuple、dict等对象时，
        # 一定要思考是否应该用ModuleList或ParameterList代替。
        # 如果你想设计一个神经网络的层数作为输入传递
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        # print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)  # permute(dims):将tensor的维度换位，[B, N, 3]
        # print(points.shape)
        # 形成局部的group
        if self.group_all:
            # new_xyz:[B, npoint, 3], new_points:[B, npoint, nsample, 3+D]
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # print(new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # 以下是pointnet操作：
        # 对局部group中每一个点做MLP操作:
        # 利用1*1的2d卷积相当于把每个group当成一个通道，共npoint个通道
        # 对[C+D，nsample]的维度上做逐像素的卷积，结果相当于对单个c+D维度做1d的卷积
        # print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # print(new_points.shape)
        # 最后进行局部的最大池化，得到局部的全局特征
        # 对每个group做一个max pooling得到局部的全局特征,得到的new_points:[B,3+D,npoint]
        new_points = torch.max(new_points, 2)[0]
        # print(new_points.shape)# new_xyz:[B, 3, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        # print(new_xyz.shape)
        return new_xyz, new_points


# PointNetSetAbstractionMsg类实现MSG方法的Set Abstraction:
# 这里radius_list输入的是一个list，例如[0.1，0.2,0.4]
# 对于不同的半径做ball query，将不同半径下的点云特征保存在new_points_list中，最后再拼接到一起
# MSG实现
class PointNetSetAbstractionMsg(nn.Module):
    '''
          PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
          Input:
              xyz: (batch_size, ndataset, 3) TF tensor
              points: (batch_size, ndataset, channel) TF tensor
              npoint: int32 -- #points sampled in farthest point sampling
              radius_list: list of float32 -- search radius in local region
              nsample_list: list of int32 -- how many points in each local region
              mlp_list: list of list of int32 -- output size for MLP on each point
          Return:
              new_xyz: (batch_size, npoint, 3) TF tensor
              new_points: (batch_size, npoint, sum_k{mlp[k][-1]}) TF tensor
      '''

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # 就是坐标点位置特征
        # print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)  ##就是额外提取的特征，第一次的时候就是那个法向量特征
        # print(points.shape)
        B, N, C = xyz.shape
        S = self.npoint
        # 最远点采样
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # 采样后的点
        print(new_xyz.shape)
        # 将不同半径下的点云特征保存在new_points_list
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # query_ball_point函数用于寻找球形邻域中的点
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # 返回的是索引
            # 按照输入的点云数据和索引返回索引的点云数据
            grouped_xyz = index_points(xyz, group_idx)  # 得到各个组中实际点
            grouped_xyz -= new_xyz.view(B, S, 1, C)  # 去mean new_xyz相当于簇的中心点
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # 拼接点云特征数据和点坐标数据
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                # print(grouped_points.shape)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            # print(grouped_points.shape)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            # print(grouped_points.shape)
            # 最大池化，获得局部区域的全局特征
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S] 就是pointnet里的maxpool操作
            # print(new_points.shape)
            new_points_list.append(new_points)  # 不同半径下的点云特征的列表

        new_xyz = new_xyz.permute(0, 2, 1)
        # 拼接不同半径下的点云特征
        new_points_concat = torch.cat(new_points_list, dim=1)
        # print(new_points_concat.shape)
        return new_xyz, new_points_concat


# Feature Propagation的实现主要通过线性差值和MLP完成
# 当点的个数只有一个的时候，采用repeat直接复制成N个点
# 当点的个数大于一个的时候，采用线性差值的方式进行上采样
# 拼接上下采样对应点的SA的特征，再对拼接后的每一个点做一个MLP
# 实现主要通过线性差值与MLP堆叠完成，距离越远的点权重越小，最后对于每一个点的权重再做一个全局的归一化
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):  # 例如in_channel=384，mlp=[256,128]
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # print(xyz1.shape)
        # print(xyz2.shape)

        points2 = points2.permute(0, 2, 1)
        # print(points2.shape)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # 当点的个数只有一个的时候，采用repeat直接复制成N个点
            interpolated_points = points2.repeat(1, N, 1)
            # print(interpolated_points.shape)
        else:
            # 当点的个数大于一个的时候，采用线性差值的方式进行上采样
            dists = square_distance(xyz1, xyz2)
            # print(dists.shape)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)  # 距离越远的点权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 对于每一个点的权重再做一个全局的归一化
            # print(weight.shape)
            # print(index_points(points2, idx).shape)
            # 获得插值点
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            # print(interpolated_points.shape)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            # 拼接上下采样前对应点SA层的特征
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        # print(new_points.shape)
        new_points = new_points.permute(0, 2, 1)
        # print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # print(new_points.shape)
        return new_points



