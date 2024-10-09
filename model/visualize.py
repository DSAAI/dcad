import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from data.testdata_multi_graph import MultiGraphTraceTestDataSet
from ocgin import *
import glob
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.manifold import MDS
from loader import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)  #将当前轴设置为ax
    return cbar

def downsampling_embeds(embeds, ys, down_rate=1):
    # 需要在排序后调用
    index_anomaly = np.where(ys==1)[0]   #获取异常点的索引

    #从异常索引中选出一部分作为下采样异常索引
    sub_index_anomaly = np.random.choice(index_anomaly, int(len(index_anomaly)*down_rate), replace=False)

    #将下采样异常索引和正常索引进行合并
    sub_index = np.concatenate([np.where(ys==0)[0], sub_index_anomaly])

    #根据获取的索引将下采样的ys 和嵌入信息得出
    ys_sub = ys[sub_index]
    embeds_sub =  embeds[:, sub_index, :]
    return embeds_sub, ys_sub

def euclidean(embeddings):
    #embeddings = embeddings - embeddings.mean(0, keepdims=True)
    distances = euclidean_distances(embeddings, embeddings, squared=False)
    distances /= distances.max() # between 0 and 1   在0和1之间
    return 1 - distances # similarity  将其作为相似度

def sort_embeds(embeds, ys):
    order = np.argsort(ys)
    ys = ys[order]
    embeds = embeds[:, order,:]
    return embeds, ys

def get_performance(embed, ys, center):
    anomaly_scores = ((embed - center)**2).sum(1)
    return roc_auc_score(ys, anomaly_scores)

def load_model_get_embedding(ckpt,data_name, dataset,batch_size = 32,   seed = 1228, Model=OCGIN, down_rate=1.0):
    torch.manual_seed(seed)
    m = Model.load_from_checkpoint(ckpt)
    m.eval()
    m.cuda()


    ys = torch.cat([data.y for data in dataset])

    #dataset应为测试集
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # pass data into model to get embeddings
    with torch.no_grad():
        embeds, ys = [], []
        for data in loader:
            # batch_embeds, batch_stds = m(data.to(device))
            embeds.append(m.get_hiddens(data.to(torch.device("cuda"))))
            ys.append(data.y)

    embeds = torch.cat(embeds, dim=1).cpu().numpy()
    embeds = np.add.accumulate(embeds, axis=0)
    ys = torch.cat(ys).cpu().numpy()

    # get centers for each accumulative embeddings
    centers = m.all_layer_centers.cpu().numpy()
    centers = np.add.accumulate(centers, axis=0)
    return embeds, ys, centers

def load_model_get_performance_perseed(data_name, down_class, second_class, nlayer, epoch, seed, Model=OCGIN, down_rate=0.1):
    embeds, ys, centers = load_model_get_embedding(data_name, down_class, second_class, nlayer, epoch, seed, Model=Model, down_rate=down_rate)
    # get performance
    rocs = []
    for ii, embed in enumerate(embeds):
        roc = get_performance(embed, ys, centers[ii])
        rocs.append(roc)
    return rocs

def load_model_get_performance(data_name, down_class, second_class, epoch, nlayer=5, mean=True, down_rate=0.1):
    seeds = [15213, 10086, 11777, 333, 444, 555, 666, 777, 888, 999]
    rocs = []
    for seed in seeds:
        rocs_seed = load_model_get_performance_perseed(data_name, down_class, second_class, nlayer, epoch, seed, down_rate=down_rate)
        rocs.append(rocs_seed)
    rocs = np.array(rocs)
    if mean is True:
        rocs = rocs.mean(axis=0)
    return rocs

def visualize_roc_vs_sampling_rate(data_name, down_class, second_class):
    # This function needs running train_ocgin_without_save first  此函数需要先运行train_ocgin_without_save
    # read results from file   从文件中读取结果
    down_rates = np.linspace(0.05,0.85,17)
    file_name = os.path.join('results', f'{data_name}-OCGIN-different_down_rate.log')
    results = {rate:{} for rate in down_rates}
    with open(file_name, 'r') as file:
        lines = filter(lambda x:x[0]=='[' or x[0]=='!' or x[0]=='{', file.readlines())
        for line in lines:
            line = line.split()
            if len(line) == 6:
                line = [x.split('[') for x in line]
                line = {x[0]: x[1].strip(']') for x in line}
                key = line['!Data'] + '-' +  line['DownClass'] + '-' + line['SecondClass']
                rate = float(line['DownRate'])
                if not key in results[rate]:
                    results[rate][key] = []
            else:
                results[rate][key].append(float(line[1].strip(',') ))
    rocs = np.zeros((2, len(down_rates), 10))
    for i, rate in enumerate(down_rates):
        rocs[0, i] = np.array(results[rate][f'{data_name}-{down_class}-{second_class}'])
        rocs[1, i] = np.array(results[rate][f'{data_name}-{second_class}-{down_class}'])
    # plot
    plt.figure()
    rocs0 = rocs[0].T
    rocs1 = rocs[1].T
    for roc in rocs0:
        plt.plot(down_rates, roc, c='red', alpha=0.2)
    plt.plot(down_rates, rocs0.mean(axis=0), c='red', label='class %d as outlier'%down_class)
    for roc in rocs1:
        plt.plot(down_rates, roc, c='blue', alpha=0.2)
    plt.plot(down_rates, rocs1.mean(axis=0), c='blue', label='class %d as outlier'%second_class)
    plt.legend()
    plt.xlabel('Outlier downsampling rate')
    plt.ylabel('Outlier detection ROC-AUC')
    plt.title('Number of layer = {}'.format(5))


def visualize_roc_vs_iteration(data_name, down_class, second_class, nlayer=5, epoch=25, down_rate=0.1):
    fig = plt.figure()
    rocs0 = load_model_get_performance(data_name, down_class, second_class, epoch, mean=False, down_rate=down_rate)
    rocs1 = load_model_get_performance(data_name, second_class, down_class, epoch, mean=False, down_rate=down_rate)
    for roc in rocs0:
        plt.plot(roc, c='red', alpha=0.2)
    plt.plot(rocs0.mean(axis=0), c='red', label='class %d as outlier'%down_class)
    for roc in rocs1:
        plt.plot(roc, c='blue', alpha=0.2)
    plt.plot(rocs1.mean(axis=0), c='blue', label='class %d as outlier'%second_class)

    plt.legend()
    plt.xlabel("Number of layers")
    plt.ylabel("Outlier detection ROC-AUC")
    plt.title("Outlier Downsampling Rate = {}".format(down_rate))


def visualize_disperity_downsampled(data_name, down_class, second_class, nlayer, epoch, seed, Model=OCGIN, gap=1):
    embeds, ys, centers = load_model_get_embedding(data_name, down_class, second_class, nlayer, epoch, seed,
                                                   Model=Model, down_rate=0.1)
    # sort embeds centers and ys
    embeds, ys = sort_embeds(embeds, ys)
    k = 20

    # get boundaries, assume data is sorted
    diff = np.roll(ys, 1, axis=0) - ys
    boundaries = np.nonzero(diff)[0]

    # create label for both class
    label0, label1, color0, color1 = 'Inlier', 'Outlier', 'blue', 'red'
    colors = np.array([color0, color1])

    embeds = np.stack([embeds[i] for i in range(len(embeds)) if i % gap == 0])
    centers = np.stack([centers[i] for i in range(len(centers)) if i % gap == 0])

    fig, axes = plt.subplots(1, len(embeds), figsize=(len(embeds) * 3, 3), constrained_layout=True)
    fig.suptitle(r'\underline{\textbf{Class\ ' + str(down_class) + '\ as\ outlier}}  with its downsampling rate=0.1',
                 fontsize=15, usetex=True)

    # get performance
    rocs = []
    for ii, embed in enumerate(embeds):
        roc = get_performance(embed, ys, centers[ii])
        rocs.append(roc)

        percentages = {0: [], 1: []}
        radiuses = {0: [], 1: []}
        # calculate similarity matrix  计算相似度矩阵
        kernel_matrix = euclidean(embed)
        # overlap counting
        for i, y in enumerate(ys):
            similarities_to_other_nodes = kernel_matrix[i]
            sort_index = np.argsort(similarities_to_other_nodes)[::-1][:k]  # descending order
            neighbors = ys[sort_index]
            percentage_of_abnormal = sum(neighbors != y) / len(neighbors)
            percentages[y].append(percentage_of_abnormal)
            radius = 1 - similarities_to_other_nodes[sort_index[-1]]
            radiuses[y].append(radius)
        ax = axes[ii]
        ax.set_title('\#layer={}, roc='.format(ii * gap) + r'\underline{\textbf{' + '{:.3f}'.format(roc) + '}}',
                     usetex=True)
        sns.distplot(percentages[0], hist=False, color=color0, label=label0, ax=ax)
        sns.distplot(percentages[1], hist=False, color=color1, label=label1, ax=ax)

        ax.set_xlabel('Disagree% in {}-NN'.format(k))
        ax.set_ylabel('Density of graphs'.format(k))
        ax.set_xlim(-0.1, 1.1)
        ax.legend()

def visualize_disperity_full(ckpt,data_name,dataset,batch_size, down_class, second_class, seed =1228, Model=OCGIN, gap=1):
    #加载模型并获取嵌入信息
    embeds, ys, centers = load_model_get_embedding(ckpt,data_name,dataset , batch_size, seed,
                                                   Model=Model, down_rate=1)

    #将函数赋值给roc_func
    roc_func = get_performance
    k = 20

    # sort embeds centers and ys
    embeds_all, ys_all = sort_embeds(embeds, ys)

    # get boundaries, assume data is sorted   沿给定轴滚动数组元素。
    diff = np.roll(ys_all, 1, axis=0) - ys_all
    boundaries = np.nonzero(diff)[0]  #返回非0元素的下标

    # 创建一个下采样版本（异常类下采样）   now create a downsampled (anomaly class downsample) version
    embeds_sub, ys_sub = downsampling_embeds(embeds_all, ys_all, 1)   #该函数用于生成下采样嵌入信息和下采样ys

    # only evaluate the one between gap   只评估间隔之间的一个
    embeds_sub = np.stack([embeds_sub[i] for i in range(len(embeds_sub)) if i % gap == 0])
    centers_sub = np.stack([centers[i] for i in range(len(centers)) if i % gap == 0])

    # create label for both class  为这两个类创建标签
    label0 = 'Class %d' % second_class    #这个是正常样本
    label1 = 'Class %d' % down_class      #这个是下采样样本
    color0 = 'tab:green'
    color1 = 'tab:orange'
    colors = np.array([color0, color1])

    offset = 0  #偏移量
    fig, axes = plt.subplots(4 - offset, len(embeds_sub), figsize=(len(embeds_sub) * 3, (4 - offset) * 3),
                             constrained_layout=True)
    # create figure   给图片指定标题和字体大小
    fig.suptitle('Full data pairwise similarity visualization', fontsize=15)

    # visualize    可视化   遍历所有下采样嵌入信息
    for ii, embed in tqdm.tqdm(enumerate(embeds_sub)):
        print(1)
        roc = roc_func(embed, ys_sub, centers_sub[ii])  #roc值在这里没有使用
        percentages = {0: [], 1: []}    #生成0和1的百分比列表
        radiuses = {0: [], 1: []}
        # calculate similarity matrix
        kernel_matrix = euclidean(embed)   #相似度
        kernel_matrix_all = euclidean(embeds_all[ii * gap])

        # overlap counting 重叠计算
        for i, y in enumerate(ys_sub):
            print(2)
            similarities_to_other_nodes = kernel_matrix[i]  #获取节点相似性
            sort_index = np.argsort(similarities_to_other_nodes)[::-1][:k]  # 进行降序排列
            neighbors = ys_sub[sort_index]       #
            percentage_of_abnormal = sum(neighbors != y) / len(neighbors)
            percentages[y].append(percentage_of_abnormal)
            radius = 1 - similarities_to_other_nodes[sort_index[-1]]
            radiuses[y].append(radius)   #将半径添加进去
        print(3)
        # plot matrix   进行绘图操作
        aa = axes[0, ii].matshow(kernel_matrix_all, cmap=plt.get_cmap('RdBu').reversed())
        colorbar(aa)
        for boundary in boundaries:   #
            axes[0, ii].axhline(y=boundary, color='green', linestyle='-')
            axes[0, ii].axvline(x=boundary, color='green', linestyle='-')
        axes[0, ii].set_title('#layer={}'.format(ii * gap))
        axes[0, ii].set_xlabel('All-graph similarity matrix')

        axes[0, ii].set_xticks([boundaries[-1]])  # len(ys_all)-boundaries[-1]
        axes[0, ii].set_xticklabels(['{}  |  {}'.format(label0, label1)])
        axes[0, ii].set_yticks([boundaries[-1]])
        axes[0, ii].set_yticklabels(['{}  |  {}'.format(label1, label0)], rotation='vertical', va="center")
        print(4)

        # mds embedding
        ys = ys.astype(int)
        print(4-1)
        mds = MDS(n_components=2, dissimilarity="precomputed",normalized_stress="auto")  #多维标度
        print(4-2)
        mds_embs = mds.fit_transform(1 - kernel_matrix)         #
        print(5)
        # mds = MDS(n_components=2)
        # tsne = TSNE(n_components=2)
        # mds_embs = tsne.fit_transform(embed.astype(np.float64))

        cs = colors[ys]
        perm = np.random.permutation(len(ys))
        axes[1, ii].scatter(mds_embs[perm, 0], mds_embs[perm, 1], c=cs[perm], s=1, alpha=0.3)
        # produce a legend with the unique colors from the scatter
        axes[1, ii].scatter(0, 0, c=color0, s=1, label=label0, alpha=0.3)
        axes[1, ii].scatter(0, 0, c=color1, s=1, label=label1, alpha=0.3)
        axes[1, ii].legend()
        axes[1, ii].set_title('MDS visualization')

        sns.distplot(radiuses[0], hist=True, kde=False, color=color0, label=label0, ax=axes[2, ii])
        sns.distplot(radiuses[1], hist=True, kde=False, color=color1, label=label1, ax=axes[2, ii])
        axes[2, ii].set_xlabel('Radius of {}-NN'.format(k))
        axes[2, ii].set_xlim(0, 1)
        axes[2, ii].set_ylabel('Number of graphs')
        axes[2, ii].legend()
        print(6)
        ax = axes[3 - offset, ii] if offset < 3 else axes[ii]
        sns.distplot(percentages[0], hist=False, color=color0, label=label0, ax=ax)
        sns.distplot(percentages[1], hist=False, color=color1, label=label1, ax=ax)
        print(7)
        ax.set_xlabel('Disagree% in {}-NN'.format(k))
        ax.set_ylabel('Density of graphs)'.format(k))
        ax.set_xlim(-0.1, 1.1)
        ax.legend()


def investigate_model_embeddings(data_name, down_class, second_class, nlayer=5, epoch=25, seed=15213, Model=OCGIN):
    # define result saving path 定义结果   保存路径
    result_dir = os.path.join('OCGIN_Plots', f'{data_name}-{down_class}-{second_class}', f'nlayer{nlayer}')
    # 如果结果目录不存在   就创建一个
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    gap =1
    # visualize full data  可视化所有数据
    visualize_disperity_full(data_name, down_class, second_class, nlayer, epoch, seed, gap=gap)

    plt.savefig(os.path.join(result_dir, 'overlap_disperity-fulldata-{}-{}-epoch{}-downcls{}.'.format(
        data_name, Model.__name__, epoch, 1 )+ format), format=format)
    plt.close()
    # visualize downsampled data (transductive)  可视化下采样数据
    visualize_disperity_downsampled(data_name, down_class, second_class, nlayer, epoch, seed, gap=gap)
    plt.savefig(os.path.join(result_dir ,'overlap_disperity-downsampled(c{}-r{})-{}-{}-epoch{}.'.format(
        down_class, 0.1, data_name, Model.__name__, epoch ) + format) ,format=format)
    plt.close()
    #可视化下采样数据
    visualize_disperity_downsampled(data_name, second_class, down_class, nlayer, epoch, seed, gap=gap)
    plt.savefig(os.path.join(result_dir ,'overlap_disperity-downsampled(c{}-r{})-{}-{}-epoch{}.'.format(
        second_class, 0.1, data_name, Model.__name__, epoch )+ format), format=format)
    plt.close()

    # visualize roc-vs-iteration  down_rate=0.1    可视化roc
    visualize_roc_vs_iteration(data_name, down_class, second_class, nlayer, epoch)
    plt.savefig \
        (os.path.join(result_dir ,'roc_vs_iter-{}-{}-epoch{}.'.format(data_name, Model.__name__ ,epoch ) +format), format=format)
    plt.close()

    # visualize roc-vs-samplingrate    可视化roc
    visualize_roc_vs_sampling_rate(data_name, down_class, second_class)
    plt.savefig \
        (os.path.join(result_dir ,'roc_vs_downrate-{}-{}-epoch{}.'.format(data_name, Model.__name__ ,epoch ) +format), format=format)
    plt.close()

if __name__ == '__main__':
    testdataset = MultiGraphTraceTestDataSet(root='data/testdata50')
    result_dir = os.path.join("/home/cjl/mynet/test",'OCGIN_Plots', f'mydata1-1-0', f'nlayer5')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    gap = 1
    # visualize full data  可视化所有数据
    visualize_disperity_full(r"OCGIN/mydata1-1-0/nlayer-5/seed-1226/timestamp=1690549618.570001_epoch=100_roc_auc=0.903.ckpt", "mydata1",testdataset, 1024, 1, 0, seed = 1228, Model = OCGIN, gap = 1)
    plt.savefig(os.path.join(result_dir, 'overlap_disperity-fulldata-{}-{}-epoch{}-seed{}-timestamp{}.'.format("mydata1", "OCGIN", 8, 1228,1690549618.570001) ))
    plt.close()


