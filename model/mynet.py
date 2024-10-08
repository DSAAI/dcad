import torch
from utils.functions import process_grahp_edge_attr, loss_func_v2, getcombineloss, loss_func, process_graph


class MyNet():
    def __init__(self,lr,epoch,batch_size,train_loader,device,model,alpha,modeltype):
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.device = device
        self.model = model
        self.alpha = alpha
        self.modeltype = modeltype
    def fit(self):
        res = []
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        num = 0
        for sampled_data in self.train_loader:
            num+=1
            epoch_loss = 0
            for i in range(0,self.batch_size):


                #print(len(x), len(edge_index[0]))


                if self.modeltype=="combinemodel":
                    x, edge_index, edge_attr, s = process_grahp_edge_attr(sampled_data, self.device, i)
                    edge_type = torch.tensor([0 for i in edge_attr], dtype=torch.long).to(self.device)
                    embedding_x,x_, s_ = self.model(x, edge_index, edge_attr = edge_attr,edge_type = edge_type)
                    structure_loss , latency_loss= loss_func_v2(embedding_x,x_ ,s, s_)
                    loss = getcombineloss(structure_loss,latency_loss,self.alpha)
                elif self.modeltype=="anomalymodel":
                    x, edge_index,  s = process_graph(sampled_data, self.device, i)
                    s_ = self.model(x, edge_index)
                    #print(s,s_)
                    loss = loss_func(s, s_)
                    #print(loss.item())
                else:
                    raise Exception("无对应类型的模型："+self.modeltype)

                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            res.append("epoch " + str(num) + ":" + str(epoch_loss / len(sampled_data.y.to(self.device))))
            print("epoch " + str(num) + ":" + str(epoch_loss / len(sampled_data.y.to(self.device))))
            if(num == self.epoch):break

        return res

    def getModel(self):
        return self.model

