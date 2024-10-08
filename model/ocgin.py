import numpy as np
import torch
import pytorch_lightning as pl  # for training wrapper
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

class DeepGLAD(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, weight_decay=5e-4, **kwargs):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, data):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        if self.current_epoch > 0:
            return self(batch), batch.y

    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0:
            # assume label 1 is anomaly and 0 is normal. (pos=anomaly, neg=normal)
            anomaly_scores = torch.cat([out[0] for out in outputs]).cpu().detach()
            ys = torch.cat([out[1] for out in outputs]).cpu().detach()
            # import pdb; pdb.set_trace()
            precision, recall, thresholds = precision_recall_curve(ys, anomaly_scores)
            roc_auc = roc_auc_score(ys, anomaly_scores)
            pr_auc = average_precision_score(ys, anomaly_scores)
            avg_score_normal = anomaly_scores[ys == 0].mean()
            avg_score_abnormal = anomaly_scores[ys == 1].mean()

            metrics = {'val_roc_auc': roc_auc,
                       'val_pr_auc': pr_auc,
                       'val_average_score_normal': avg_score_normal,
                       'val_average_score_anomaly': avg_score_abnormal,"precision_recall":max(precision)}

            self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        return self(batch), batch.y  # .squeeze(-1)

    def test_epoch_end(self, outputs):
        # assume label 1 is anomaly and 0 is normal. (pos=anomaly, neg=normal)
        anomaly_scores = torch.cat([out[0] for out in outputs]).cpu().detach()
        ys = torch.cat([out[1] for out in outputs]).cpu().detach()
        # import pdb; pdb.set_trace()
        precision, recall, thresholds = precision_recall_curve(ys,anomaly_scores)
        roc_auc = roc_auc_score(ys, anomaly_scores)
        pr_auc = average_precision_score(ys, anomaly_scores)
        avg_score_normal = anomaly_scores[ys == 0].mean()
        avg_score_abnormal = anomaly_scores[ys == 1].mean()
        max_f1 = 0
        f1_idx = 0

        for i, pre in enumerate(precision):

            rec = recall[i]
            temp = 2*pre*rec / (rec+pre)
            if(temp >max_f1):
                max_f1 = temp
                f1_idx = i

        metrics = {'roc_auc': roc_auc,
                   'pr_auc': pr_auc,
                   'average_score_normal': avg_score_normal,
                   'average_score_anomaly': avg_score_abnormal,
                   "precision_":precision[f1_idx],
                   "recall":recall[f1_idx],
                   "f1":max_f1
                   }

        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)



import torch.nn as nn
from torch.nn import BatchNorm1d,BatchNorm2d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(nn.Module):

    """
    Note: batch normalization can prevent divergence maybe, take care of this later.
    """

    def __init__(self, nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))  #从某种角度将 这里更像是对节点信息进行规格化
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):

            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias),act, Linear(nhid, nhid, bias=bias)))

            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #
        #print(len(x),len(x[0]),len(edge_index[0]))

        #x = self.item_embedding(x).squeeze(1)
        #print(x.shape, edge_index.shape)
        #print(x.shape)
        #print("before_x:", x.shape,x)#before_x: torch.Size([851, 22])
        x = self.transform(x)
        #print("after_x:",x.shape, x)#after_x: torch.Size([851, 128])
        # weird as normalization is applying to all ndoes in database
        # maybe a better way is to normalize the mean of each graph, and then apply tranformation
        # to each groups *
        #print("pooling:",x.shape,batch.shape)
        #after_batch = torch.tensor(np.zeros((len(batch),22,128)))

        embed = self.pooling(x, batch)
        #print("embed:", embed.shape,embed)#embed: torch.Size([128, 128])
        #print("batch", batch.shape,batch)#batch: torch.Size([851])
        #print("embed",embed)
        #print("embed[batch]:",embed[batch].shape,embed[batch])#embed[batch]: torch.Size([851, 128])
        std = torch.sqrt(self.pooling((x - embed[batch]) ** 2, batch))
        #print("std:", std.shape,std)

        graph_embeds = [embed]
        graph_stds = [std]
        #
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)
            embed = self.pooling(x, batch)  # embed is the center of nodes
            std = torch.sqrt(self.pooling((x - embed[batch]) ** 2, batch))
            graph_embeds.append(embed)
            graph_stds.append(std)
        #print("before_graph_embeds",graph_embeds)
        graph_embeds = torch.stack(graph_embeds)  #对嵌入信息进行拼接
        #print("after_graph_embeds",graph_embeds)
        graph_stds = torch.stack(graph_stds)
        return graph_embeds, graph_stds


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool


class OCGIN(DeepGLAD):
    def __init__(self, nfeat,
                 nhid=128,
                 nlayer=3,
                 dropout=0,
                 learning_rate=0.001,
                 weight_decay=0,
                 **kwargs):
        model = GIN(nfeat, nhid, nlayer=nlayer, dropout=dropout)
        super().__init__(model, learning_rate, weight_decay)
        self.save_hyperparameters()  # self.hparams
        self.radius = 0
        self.nu = 1
        self.eps = 0.01
        self.mode = 'sum'
        assert self.mode in ['concat', 'sum']
        self.register_buffer('center', torch.zeros(nhid if self.mode == 'sum' else (nlayer + 1) * nhid))
        self.register_buffer('all_layer_centers', torch.zeros(nlayer + 1, nhid))

    def get_hiddens(self, data):
        embs, stds = self.model(data)
        return embs

    def forward(self, data):
        embs, stds = self.model(data)
        if self.mode == 'concat':
            embs = torch.cat([emb for emb in embs], dim=-1)
        else:
            # sum is used by original GIN method
            embs = embs.sum(dim=0)
        #self.center是一个128维的数组  维数由nhid决定

        dist = torch.sum((embs - self.center) ** 2, dim=1)  #图级嵌入信息到center的距离
        scores = dist - self.radius ** 2  #距离减去容忍半径


        return scores

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            # init the center (and radius)   初始化中心和半径
            #print(batch)
            embs, stds = self.model(batch)  # for all nodes in the batch
            loss = torch.zeros(1, requires_grad=True, device=self.device)  # don't update
            return {'loss': loss, 'emb': embs.detach()}
        else:
            assert self.center != None
            scores = self(batch)        #等价于调用该类的forward方法
            loss = self.radius ** 2 + (1 / self.nu) * torch.mean(F.relu(scores))   #容忍半径加上图评分的平均值

            self.log('training_loss', loss)

            return loss

    def training_epoch_end(self, outputs):
        #对self.center进行更新
        if self.current_epoch == 0:
            # 初始化center
            embs = torch.cat([d['emb'] for d in outputs], dim=1)   #将所有的outputs输出的tensor按维度1进行拼接
            self.all_layer_centers = embs.mean(dim=1)
            if self.mode == 'concat':
                self.center = torch.cat([x for x in self.all_layer_centers], dim=-1)
            else:
                # sum is used by original GIN method
                self.center = torch.sum(self.all_layer_centers, 0)

            # self.register_buffer('center', center)
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            # self.center[(abs(self.center) < self.eps) & (self.center < 0)] = -self.eps
            # self.center[(abs(self.center) < self.eps) & (self.center > 0)] = self.eps


