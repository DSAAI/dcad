import torch
from torch_geometric.utils import to_dense_adj

def loss_func( s, s_):

    #print(type(s),type(s_))
    # structure reconstruction loss
    diff_structure = torch.pow(s - s_, 2)
    structure_score = torch.sqrt(torch.sum(diff_structure, 1))
    structure_loss = torch.sum(structure_score)
    return structure_loss

def loss_func_v2(x,x_,s,s_):

    diff_latency = torch.pow(x - x_, 2)
    latency_score = torch.sqrt(torch.sum(diff_latency, 1))
    latency_loss = torch.sum(latency_score)
    latency_loss = torch.mean(latency_loss)
    # structure reconstruction loss
    #print(type(s), type(s_))
    diff_structure = torch.pow(s - s_, 2)
    structure_score = torch.sqrt(torch.sum(diff_structure, 1))
    structure_loss = torch.sum(structure_score)
    structure_loss = torch.mean(structure_loss)
    return structure_loss, latency_loss


def getcombineloss(structure_loss,latency_loss,alpha):

    return alpha * latency_loss \
            + (1 - alpha) * structure_loss

def process_graph(G,device,i):
    x, edge_index = G[i].x.to(device), G[i].edge_index.to(device)
    #print(x,edge_index)
    s = to_dense_adj(edge_index)[0].to(device)
    return x,edge_index,s

def process_grahp_edge_attr(G,device,i):
    x, edge_index, edge_attr = G[i].x.to(device), G[i].edge_index.to(device) , G[i].edge_attr.to(device)
    # print(x,edge_index)
    s = to_dense_adj(edge_index)[0].to(device)
    return x, edge_index,edge_attr, s


def decision_function(model,train_loader,device):

    model.eval()
    outlier_scores = []
    for sampled_data in train_loader:
        x,  edge_index ,s= process_graph(sampled_data,device)
        s_ = model(x, edge_index)
        score = loss_func(s,s_)
        outlier_scores.append(score.item())
    return outlier_scores


def standard_edge(edge_group):

    '''
    rawNetworklatency = edge_group.rawNetworklatency.values  # 未加工的网络延迟
    rawProcessingTime = edge_group.rawProcessingTime.values  # 未加工的处理时长
    proportionProcessingTime = edge_group.proportionProcessingTime.values  # 处理时长占比
    rawDuration = edge_group.rawDuration.values  # 未处理的时长信息
    isError = edge_group.isError.values  # 该次请求是否出现错误
    workDuration = edge_group.workDuration.values  # 工作时长
    statusCode = edge_group.statusCode.values  # 状态码
    duration = edge_group.duration.values  # 总时长
    :param edge_group:
    :return:
    '''
    #22  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    res = []
    res.append(int(edge_group.rawNetworklatency.min()))
    res.append(int((edge_group.rawNetworklatency.max())))
    res.append(int(edge_group.rawNetworklatency.mean()))
    res.append(int(edge_group.rawProcessingTime.min()))
    res.append(int(edge_group.rawProcessingTime.max()))
    res.append(int(edge_group.rawProcessingTime.mean()))
    res.append(int((edge_group.proportionProcessingTime.min()) * 1000))
    res.append(int((edge_group.proportionProcessingTime.max())* 1000))
    res.append(int((edge_group.proportionProcessingTime.mean()) * 1000))
    res.append(int(edge_group.rawDuration.min()))
    res.append(int(edge_group.rawDuration.max()))
    res.append(int(edge_group.rawDuration.mean()))
    res.append(int(edge_group.workDuration.min()))
    res.append(int(edge_group.workDuration.max()))
    res.append(int(edge_group.workDuration.mean()))
    res.append(int((edge_group.duration.min()) * 10000))
    res.append(int((edge_group.duration.max()) * 10000))
    res.append(int((edge_group.duration.mean()) * 10000))
    t = 0
    f = 0
    for e in edge_group.isError.values:
        if(e == "FALSE"):
            f+=1
        else:
            t+=1
    res.append(t)
    res.append(f)
    t = 0
    f = 0
    for e in edge_group.statusCode.values:
        if(e == "5"):
            t+=1
        else:
            f+=1
    res.append(t)
    res.append(f)
    return res

def makefilename(time,learn_rate,train_epoch,batch_size,device,modeltype):
    return time+"-"+str(learn_rate)+"-"+str(train_epoch)+"-"+str(batch_size)+"-"+str(device)+"-"+str(modeltype)

