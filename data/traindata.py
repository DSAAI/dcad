import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data
import torch
from tqdm import tqdm


class TraceTrainDataSet(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super(TraceTrainDataSet, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['train.dataset']

    def download(self):
        pass
    def process(self):
        csv_file = "/home/cjl/csv/train.csv"
        df = pd.read_csv(csv_file)
        data_list = []
        grouped = df.groupby('traceId')
        for traceId,group in tqdm(grouped):
            #group = group.drop_duplicates(subset='operationId')
            group = group.reset_index(drop=True)
            max_target = group.target.max()
            max_source = group.source.max()
            label = group.abnormal.max()

            edge_group = group.drop_duplicates(subset='operationId')
            rawNetworklatency = edge_group.rawNetworklatency.values
            rawProcessingTime = edge_group.rawNetworklatency.values
            proportionProcessingTime = edge_group.proportionProcessingTime.values
            rawDuration = edge_group.rawDuration.values
            isError = edge_group.isError.values
            workDuration = edge_group.workDuration.values
            statusCode = edge_group.statusCode.values
            duration = edge_group.duration.values

            edge_feature = []
            for element in range(0,len(rawNetworklatency)):
                edge_feature.append([rawNetworklatency[element], rawProcessingTime[element], proportionProcessingTime[element], rawDuration[element], isError[element], workDuration[element],
                 statusCode[element], duration[element]])
            edge_features = torch.tensor(edge_feature, dtype=torch.float32)

            #node_features = group.loc[group.traceId == traceId,['target']].values
            #node_features = torch.LongTensor(np.insert(node_features,0,0,axis=0)).squeeze(1)

            sources = group.source.values
            targets = group.target.values
            source_service_id = group.source_service_id.values
            target_service_id = group.target_service_id.values
            features = []
            service_dict = {}
            for i in range(0,len(sources)):
                service_dict[sources[i]] = source_service_id[i]
                service_dict[targets[i]] = target_service_id[i]

            for i in range(0,max(max_target,max_source)+1):
                features.append(service_dict[i])

            node_features = torch.tensor(features, dtype=torch.int64)

            duplicated_group = group.drop_duplicates(subset='operationId')
            source = duplicated_group.source.values
            target = duplicated_group.target.values
            edge_index = torch.tensor([source, target], dtype=torch.long)
            if label == 1:
                y = torch.FloatTensor(np.ones(max(max_target, max_source) + 1, dtype=np.float32))
            else:
                y = torch.FloatTensor(np.zeros(max(max_target, max_source) + 1,dtype=np.float32))
            #y = torch.FloatTensor(np.insert(group.abnormal.values, 0, 0, axis=0))
            #y = torch.FloatTensor(np.insert(group.abnormal.values, 0, 0, axis=0))
            #y = torch.FloatTensor([group.abnormal.values[0]])
            data = Data(x=node_features , edge_index=edge_index ,edge_attr= edge_features,y=y)#,edge_attr= edge_features
            data_list.append(data)

        data,slices = self.collate(data_list)

        torch.save((data,slices),self.processed_paths[0])

