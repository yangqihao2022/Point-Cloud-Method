# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    # 计算pc簇的中心点，新的中心点每一个特征的值，是该簇所有数据在该特征的平均值
    centroid = np.mean(pc, axis=0)
    # 3D数据簇减去中心得到到中心的绝对距离
    pc = pc - centroid
    # 取到最大距离
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 将数据归一化
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints  # 采样点数
        self.root = root  # 文件根路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 类别和文件夹名字对应的路径
        self.cat = {}
        self.normal_channel = normal_channel  # 是否使用rgb信息


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:    # 选择一些类别进行训练
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}    # 读取分好类的文件夹jason文件 并将他们的名字放入列表中
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
                # 生成一个字典，将类别名字和训练的路径组合起来  作为一个大类中符合训练的数据
                # 上面的代码执行完之后，就实现了将所有需要训练或验证的数据放入了一个字典中，字典的键是该数据所属的类别，例如飞机。值是他对应数据的全部路径

        self.datapath = []
        for item in self.cat:      # self.cat 是类别名称和文件夹对应的字典
            for fn in self.meta[item]:
                self.datapath.append((item, fn))  # 生成标签和点云路径的元组， 将self.met 中的字典转换成了一个元组

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i] # self.classes  将类别的名称和索引对应起来

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
               shapenet 有16 个大类，然后每个大类有一些部件 ，例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1  2 3 的四个小类都属于飞机这个大类
               self.seg_classes 就是将大类和小类对应起来
        """
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:  # 初始slef.cache为一个空字典，这个的作用是用来存放取到的数据，并按照(point_set, cls, seg)放好 同时避免重复采样
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]  # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）
            cat = self.datapath[index][0]  # 拿到类名
            cls = self.classes[cat]  # 将类名转换为索引
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)  # size 20488,7 读入这个txt文件，共20488个点，每个点xyz rgb +小类别的标签
            if not self.normal_channel:  # 判断是否使用rgb信息
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)  # 拿到小类别的标签
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 做一个归一化

        choice = np.random.choice(len(seg), self.npoints, replace=True)  # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        point_set = point_set[choice, :]  # 根据索引采样
        seg = seg[choice]

        return point_set, cls, seg  # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return len(self.datapath)



