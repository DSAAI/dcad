from torch_geometric.data import DataLoader
import numpy as np
import torch, os
from torch_geometric.utils import degree
DATA_PATH = 'datasets'

import torch.nn.functional as F
class OneHotDegree(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = truncate_degree(degree(idx, data.num_nodes, dtype=torch.long))
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_degree)


def truncate_degree(degree):
    degree[ (100<=degree) & (degree <200) ] = 101
    degree[ (200<=degree) & (degree <500) ] = 102
    degree[ (500<=degree) & (degree <1000) ] = 103
    degree[ (1000<=degree) & (degree <2000) ] = 104
    degree[ degree >= 2000] = 105
    return degree



def load_data(data_name, down_class=0, down_rate=1, second_class=None, seed=1213, return_raw=False):
    ignore_edge_weight =True   #忽略边权重
    one_class_train = False     #一类训练
    np.random.seed(seed)        #随机数种子
    torch.manual_seed(seed)     #设置随机数种子  这里有使用么？

    if data_name in ['MNIST', 'CIFAR10']:
        dataset_raw = GNNBenchmarkDataset(root=DATA_PATH, name=data_name)
    else:
        # TUDataset
        use_node_attr = True if data_name == 'FRANKENSTEIN' else False
        dataset_raw = TUDataset(root=DATA_PATH, name=data_name, use_node_attr=use_node_attr)

    if return_raw:   #是否返回未经处理的数据集
        return dataset_raw

    # downsampling  显示数据信息
    # Get min and max node and filter them,
    num_nodes_graphs = [data.num_nodes for data in dataset_raw]   #记录每个图种的节点数
    min_nodes, max_nodes = min(num_nodes_graphs), max(num_nodes_graphs)    #获得图的最大节点数和最小节点数
    if max_nodes >= 10000:      #如果节点数太大就改为  10000
        max_nodes = 10000
    print("min nodes, max nodes:", min_nodes, max_nodes)

    # build the filter and transform the dataset
    filter = DownsamplingFilter(min_nodes, max_nodes, down_class, down_rate, dataset_raw.num_classes, second_class)  #将数据集进行调整  包括图大小  图的种类
    indices = [i for i, data in enumerate(dataset_raw) if filter(data)]
    # now down_class is labeled 1, second_class is labeled 0
    dataset = dataset_raw[torch.tensor(indices)].shuffle() # shuffle the dataset  将数据集打乱顺序

    # report the proportion info of the dataset
    labels = np.array([data.y.item() for data in dataset])
    label_dist = ['%d '% (labels == c).sum() for c in range(dataset.num_classes)]
    print("Dataset: %s, Number of graphs: %s [orignal classes: %d, %d], Num of Features %d" %(
        data_name, label_dist, second_class if second_class is not None else 1- down_class,
        down_class, dataset.num_features))

    # preprocessing: do not use original edge features or weights
    if ignore_edge_weight:
        dataset.data.edge_attr = None

    # deal with no-attribute case  处理数据集不包含属性的情况
    """
    Another way: discreteize the degree by range. Not always one hot. 
    """
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset_raw:  # ATTENTION: use dataset_raw instead of downsampled version!
            degs += [truncate_degree(degree(data.edge_index[0], dtype=torch.long))]
            max_degree = max(max_degree, degs[-1].max().item())
        dataset.transform = OneHotDegree(max_degree)

    # now let's transform in memory before feed into dataloader to save runtime
    dataset_list = [data for data in dataset]

    n = (len(dataset) + 9) // 10
    m = 9
    train_dataset = dataset_list[:m * n]  # 90% train
    val_dataset = dataset_list[m * n:]
    test_dataset = dataset_list

    if one_class_train:
        indices = [i for i, data in enumerate(train_dataset) if data.y.item() == 0]
        train_dataset = train_dataset[torch.tensor(indices)]  # only keep normal class left

    return train_dataset, val_dataset, test_dataset, dataset, max_nodes




def create_loaders(data_name, batch_size=32, down_class=0, second_class=None, down_rate=1, seed=15213, num_workers=0):

    train_dataset, val_dataset, test_dataset, dataset, max_nodes = load_data(data_name,
                                                                                down_class=down_class,
                                                                                second_class=second_class,
                                                                                down_rate=down_rate,
                                                                                seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset, max_nodes












