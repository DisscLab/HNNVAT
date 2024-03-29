"""
build the gene-gene adj matrix
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
parser = argparse.ArgumentParser()
BaronMouse/Filtered_MousePancreas_data.csv', help="directory of input expr matrix")
parser.add_argument('--networkPath', type = str,
                    default = 'C:/Users/Lenovo/Desktop/wkw/scDatasets/Zhengsorted/preprocessing_Zhengsorted.csv')
parser.add_argument('--dataset', type = str,
                    default = 'Zhengsorted', help = 'which dataset to load')
parser.add_argument('--net', type = str, 
                    default = 'String', help = 'which network to use: STRING or BioGrid')
parser.add_argument('--pathToSave', type = str,
                    default = 'C:/Users/Lenovo/Desktop/wkw/scDatasets/Zhengsorted/', help = 'directory to save adj matrix')

args = parser.parse_args()

def build_adj_weight(idx_features, net_filepath):

    edges_unordered =  pd.read_csv(net_filepath, index_col = None, usecols = [1,2,16])

    edges_unordered = np.asarray(edges_unordered)
    
    idx = []
    mapped_index = idx_features.index.str.upper() 
    for i in range(len(edges_unordered)): 
        if edges_unordered[i,0] in mapped_index and edges_unordered[i,1] in mapped_index: 
            idx.append(i)
    edges_unordered = edges_unordered[idx]
    print ('idx_num:',len(idx))
    del i,idx
    
    # build graph
    idx = np.array(mapped_index)
    idx_map = {j: i for i, j in enumerate(idx)} # eg: {'TSPAN12': 0, 'TSHZ1': 1}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    edges = np.array(list(map(idx_map.get, edges_unordered[:,0:2].flatten())),
                     dtype=np.int32).reshape(edges_unordered[:,0:2].shape) #map：map(function, element):function on element. 
    
    adj = sp.coo_matrix((edges_unordered[:, 2], (edges[:, 0], edges[:, 1])),
                    shape=(idx_features.shape[0], idx_features.shape[0]),
                    dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
   
    return adj

def getAdjByBiogrid(idx_features, pathnet): 
    edges_unordered =  pd.read_table(pathnet ,index_col=None, usecols = [7,8] )
    edges_unordered = np.asarray(edges_unordered)  
    
    idx = []
    for i in range(len(edges_unordered)):
        if edges_unordered[i,0] in idx_features.index and edges_unordered[i,1] in idx_features.index:
            idx.append(i)
    edges_unordered = edges_unordered[idx]
    del i,idx
    
    # build graph
    idx = np.array(idx_features.index)
    idx_map = {j: i for i, j in enumerate(idx)}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape) #map：map(function, element):function on element
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx_features.shape[0], idx_features.shape[0]),
                        dtype=np.float32)
    del idx,idx_map,edges_unordered
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#    adj = adj + sp.eye(adj.shape[0])
           
    return adj

def removeZeroAdj(adj, gedata): 
    if adj[0,0] != 0:
        #adj = adj - sp.eye(adj.shape[0])
        adj.setdiag(0)
    indd = np.where(np.sum(adj, axis=1) != 0)[0]
    adj = adj[indd, :][:, indd]
    gedata = gedata[indd,:]

   
    return adj, gedata


if __name__ == "__main__":
    network_path = args.networkPath 

    data = pd.read_csv(network_path, index_col = 0, header = 0, nrows = 5).T  # data: [gene, cell] matrix
    adj = build_adj_weight(data, network_path)
    # adj = getAdjByBiogrid(data, network_path)

    print(data.shape[0])
    sp.save_npz((args.pathToSave + '/adj'+ args.net + args.dataset + '_'+str(data.shape[0])+'.npz' ), adj)