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
parser.add_argument('--dataset', type=str,default='BaronHuman', help="dataset")
parser.add_argument('--dirAdj', type = str,default = 'C:/Users/Lenovo/Desktop/wkw/scDatasets/BaronHuman/', help = 'directory of adj matrix')
parser.add_argument('--dirLabel', type = str,default = 'C:/Users/Lenovo/Desktop/wkw/scDatasets', help = 'directory of adj matrix')
parser.add_argument('--roc_directory', type=str,default = 'roc_result/macro')
parser.add_argument('--confusion_directory', type=str,default = 'confusion_result')
parser.add_argument('--tSNE_directory', type=str,default = 'tSNE_result')
parser.add_argument('--umap_directory2', type=str,default = 'umap/gcn')
parser.add_argument('--umap_directory1', type=str,default = 'umap/raw')

parser.add_argument('--normalized_laplacian', type=bool,default = True, help='Graph Laplacian: normalized.')
parser.add_argument('--lr', type=float, default = 0.15  , help='learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--epochs', type=int, default = 80, help='# of epoch')
parser.add_argument('--batchsize', type=int, default = 64, help='# of genes')
parser.add_argument('---dropout', type=float, default = 0.3, help='dropout value')
parser.add_argument('--id1', type=str, default = '', help='test in pancreas')
parser.add_argument('--id2', type=str, default = '', help='test in pancreas')
parser.add_argument('--net', type=str, default='String', help="netWork")
parser.add_argument('--dist', type=str, default='', help="dist type")
parser.add_argument('--sampling_rate', type=float, default = 1, help='# sampling rate of cells')
parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch (default: 50)')
args = parser.parse_args()

t_start = time.process_time()

print('load data...')

adjall, alldata, labels, shuffle_index,reverse = load_largesc(path = args.dirData, dirAdj=args.dirAdj, dataset=args.dataset, net='String')
real_lable = list(reverse.values()) 


if not(shuffle_index.all()):
     shuffle_index = shuffle_index.astype(np.int32)
else:
    shuffle_index = np.random.permutation(alldata.shape[0])
    np.savetxt(args.dirData +'/' + args.dataset +'/shuffle_index_'+args.dataset+'.txt')

train_all_data, adj = down_genes(alldata, adjall, args.num_gene)

L = [laplacian(adj, normalized=True)]

train_data, val_data, test_data, train_labels, val_labels, test_labels = spilt_dataset(train_all_data, labels, shuffle_index)
args.nclass = len(np.unique(labels))  
args.train_size = train_data.shape[0]   

test_labels_re = np.array(list(map(reverse.get, test_labels)))


train_loader, val_loader, test_loader = generate_loader(train_data,val_data, test_data,
                                                        train_labels, val_labels, test_labels, 
                                                        args.batchsize)
try:
    del net
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')

net, t_total_train = train_model(Graph_GCN, train_loader,val_loader, L, args)


val_acc,confusionGCN, predictions, preds_labels, y_true, t_total_test = test_model(net, val_loader, L, args)
print('  accuracy(val) = %.3f , time= %.3f' % (val_acc, t_total_test))

test_acc,confusionGCN, predictions, preds_labels, y_true, t_total_test = test_model(net, test_loader, L, args)
print('  accuracy(test) = %.3f , time= %.3f' % (test_acc, t_total_test))

n_classes = predictions.shape[1]

data_one_hot = encode_onehot(y_true,n_classes) 

score_normed = np.array(predictions)
data_one_hot = np.array(data_one_hot)
confusionGCN = np.array(confusionGCN)

calculation(predictions, data_one_hot, preds_labels, y_true) 

preds_labels = np.array(list(map(reverse.get, preds_labels)))


 
