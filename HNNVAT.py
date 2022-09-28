import argparse
from lib.layermodel import *
from lib.utilsdata import *
from train import *
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, 'lib/')
from lib.roc import *
import numpy as np

if torch.cuda.is_available():
    # print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    # torch.cuda.manual_seed(1)
    # 重新确定种子
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dirData', type=str,default='C:/Users/Lenovo/Desktop/wkw/scDatasets/', help="directory of cell x gene matrix")
parser.add_argument('--dataset', type=str,default='Zheng68K', help="dataset")
parser.add_argument('--dirAdj', type = str,default = 'C:/Users/Lenovo/Desktop/wkw/scDatasets/Zheng68K/', help = 'directory of adj matrix')
parser.add_argument('--dirLabel', type = str,default = 'C:/Users/Lenovo/Desktop/wkw/scDatasets', help = 'directory of adj matrix')
parser.add_argument('--roc_directory', type=str,default = 'roc_result/macro')
parser.add_argument('--confusion_directory', type=str,default = 'confusion_result')
parser.add_argument('--tSNE_directory', type=str,default = 'tSNE_result')
parser.add_argument('--umap_directory2', type=str,default = 'umap/gcn')
parser.add_argument('--umap_directory1', type=str,default = 'umap/raw')

parser.add_argument('--normalized_laplacian', type=bool,default = True, help='Graph Laplacian: normalized.')
parser.add_argument('--lr', type=float, default = 0.011, help='learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--epochs', type=int, default = 40, help='# of epoch')
parser.add_argument('--batchsize', type=int, default = 64, help='# of genes')
parser.add_argument('---dropout', type=float, default = 0.3, help='dropout value')
parser.add_argument('--id1', type=str, default = '', help='test in pancreas')
parser.add_argument('--id2', type=str, default = '', help='test in pancreas')
parser.add_argument('--net', type=str, default='String', help="netWork")
parser.add_argument('--dist', type=str, default='', help="dist type")
parser.add_argument('--sampling_rate', type=float, default = 1, help='# sampling rate of cells')
parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch (default: 50)')
args = parser.parse_args()

# 原始数据聚类可视化,m没有读取和可视化
# row_plot_umap(args.dirData, args.dataset, args.umap_directory1,  args.dataset)
t_start = time.process_time()

# Load data
print('load data...')

adjall, alldata, labels, shuffle_index,reverse = load_largesc(path = args.dirData, dirAdj=args.dirAdj, dataset=args.dataset, net='String')
real_lable = list(reverse.values()) # 真实标签

# 预处理可视化
# row_plot_umap2(args.dirData, args.dataset, args.umap_directory1,  args.dataset)

if not(shuffle_index.all()):
     shuffle_index = shuffle_index.astype(np.int32)
else:
    shuffle_index = np.random.permutation(alldata.shape[0])
    np.savetxt(args.dirData +'/' + args.dataset +'/shuffle_index_'+args.dataset+'.txt')

alldata = alldata.astype(np.float32)
# down gene 选择1000个gene (1886,1000) (1000,1000)
train_all_data, adj = down_genes(alldata, adjall, args.num_gene)

L = [laplacian(adj, normalized=True)]

# 数据集拆分:8,1,1
train_data, val_data, test_data, train_labels, val_labels, test_labels = spilt_dataset(train_all_data, labels, shuffle_index)
args.nclass = len(np.unique(labels))  # 类别数
args.train_size = train_data.shape[0]   # 训练集cells

# 对test处理前的可视化
test_labels_re = np.array(list(map(reverse.get, test_labels)))
# test_plot_umap(test_data, test_labels_re, args.umap_directory1,  args.dataset)

# 加载器： 输入：数据 lable和batch
# 输出：合并数据 把数据放入DataLoader划分数据(合并的数据没变)
train_loader, val_loader, test_loader = generate_loader(train_data,val_data, test_data,
                                                        train_labels, val_labels, test_labels, 
                                                        args.batchsize)
# Delete existing network if exists
try:
    del net
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')

# 训练 输入（数据+labels）包装后的结果； 输出训练的net 和 总train time
net, t_total_train = train_model(Graph_GCN, train_loader,val_loader, L, args)


## Val 输出：acc、评估矩阵、 预测值(189,13)、 预测标签(189,)、 time
val_acc,confusionGCN, predictions, preds_labels, y_true, t_total_test = test_model(net, val_loader, L, args)
print('  accuracy(val) = %.3f , time= %.3f' % (val_acc, t_total_test))

# Test
test_acc,confusionGCN, predictions, preds_labels, y_true, t_total_test = test_model(net, test_loader, L, args)
print('  accuracy(test) = %.3f , time= %.3f' % (test_acc, t_total_test))

n_classes = predictions.shape[1]

data_one_hot = encode_onehot(y_true,n_classes) # 预测0.1比例的test

score_normed = np.array(predictions)
data_one_hot = np.array(data_one_hot)
confusionGCN = np.array(confusionGCN)

calculation(predictions, data_one_hot, preds_labels, y_true) # 去除第一行后 归一化 用one-hot做预测  当score
# preds_labels预测标签     predictions预测值(189,14)  iloc[:,0] 取第0列是truelable  变为(189,)

plot_roc_curves(n_classes, real_lable, data_one_hot, score_normed, auc, args.roc_directory, args.dataset)
plot_conf_curves(confusionGCN, args.confusion_directory, args.dataset)
# plot_tSNE(predictions, preds_labels, n_classes, args.tSNE_directory, args.dataset)

# 反转标签
preds_labels = np.array(list(map(reverse.get, preds_labels)))

plot_umap(predictions, preds_labels, args.umap_directory2,  args.dataset)


 
