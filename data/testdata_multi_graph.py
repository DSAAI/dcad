import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data
import torch
from tqdm import tqdm

from utils.functions import standard_edge


class MultiGraphTraceTestDataSet(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super(MultiGraphTraceTestDataSet, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []
    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['multi_graph_test50.dataset']

    def download(self):
        pass

    def process(self):
        csv_file = "/home/cjl/csv/testdata50.csv"
        df = pd.read_csv(csv_file)
        data_list = []
        grouped = df.groupby('traceId')
        for traceId, group in tqdm(grouped):
            # group = group.drop_duplicates(subset='operationId')
            group = group.reset_index(drop=True)
            max_target = group.target.max()
            max_source = group.source.max()
            label = group.abnormal.max()

            sources = []
            targets = []

            # [     ]长度固定   表示多条边   表示各种数据类型
            # 范围型数据   (最小值，最大值，平均值，众数)
            # 二分类数据     (值为True的个数，值为False的个数)
            # 五分类数据    (值为1xx的个数，值为2xx的个数，值为3xx的个数，值为4xx的个数，值为5xx的个数)
            edge_groups = group.groupby('operationId')  # 通过去重的方式将多条边转化为一条边
            for operationId, edge_group in edge_groups:
                sources.append(edge_group.source.min())
                targets.append(edge_group.target.min())

            edge_index = torch.tensor([sources, targets], dtype=torch.long)

            # node_features = group.loc[group.traceId == traceId,['target']].values
            # node_features = torch.LongTensor(np.insert(node_features,0,0,axis=0)).squeeze(1)

            features = []
            request_groups = group.groupby('source')
            node_features_dict = {}

            for source, request_group in request_groups:
                res = standard_edge(request_group)
                node_features_dict[source] = res
            for i in range(0, max(max_target, max_source) + 1):
                if (node_features_dict.get(i, None) != None):
                    features.append(node_features_dict[i])
                else:
                    features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            node_features = torch.tensor(features, dtype=torch.float32)

            if label == 1:
                y = torch.FloatTensor(np.ones(1, dtype=np.float32))
            else:
                y = torch.FloatTensor(np.zeros(1, dtype=np.float32))
            # y = torch.FloatTensor(np.insert(group.abnormal.values, 0, 0, axis=0))
            # y = torch.FloatTensor(np.insert(group.abnormal.values, 0, 0, axis=0))
            # y = torch.FloatTensor([group.abnormal.values[0]])
            data = Data(x=node_features, edge_index=edge_index, y=y)  # ,edge_attr= edge_features
            data_list.append(data)

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


