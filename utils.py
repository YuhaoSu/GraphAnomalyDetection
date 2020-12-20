import pickle as pkl
import scipy.io as sio
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset == 'citeseer':
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = np.array(features.todense())
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj, features


def load_other_data(dataset_address):
    a = np.load("data/{}.npz".format(dataset_address))
    features = sp.csr_matrix((a['attr_data'], a['attr_indices'], a['attr_indptr']),
                             a['attr_shape']).toarray()
    adj = sp.csr_matrix((a['adj_data'], a['adj_indices'], a['adj_indptr']),
                        a['adj_shape']).toarray()
    adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
    return adj, features


def anomaly_injection_adj(adj, clique_size, num_clique):
    if isinstance(adj, sp.csr.csr_matrix) is True:
        adj = np.array(adj.todense())
        num_nodes = adj.shape[0]
        nodes_set = np.arange(num_nodes)
        temp_nodes_set = set(np.copy(nodes_set))
        anomaly_nodes_set = []
        for i in range(num_clique):
            anomaly_nodes = np.random.choice(list(temp_nodes_set), clique_size, replace=False)
            for j in range(clique_size):
                for k in range(clique_size):
                    adj[anomaly_nodes[j], anomaly_nodes[k]] = 1
            temp_nodes_set = temp_nodes_set - set(anomaly_nodes)
            anomaly_nodes_set.append(list(anomaly_nodes))
        str_anomaly_nodes_set = [j for i in anomaly_nodes_set for j in i]
        str_normal_nodes_set = list(temp_nodes_set)
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
        return str_anomaly_nodes_set, str_normal_nodes_set, adj
    else:
        raise Exception("data type not match")


def anomaly_injection_features(features, str_anomaly_nodes_set, str_normal_nodes_set, clique_size, num_clique, k):
    if isinstance(features, np.ndarray) is True:
        num_nodes = features.shape[0]
        feat_anomaly_nodes_set = np.random.choice(list(str_normal_nodes_set), clique_size * num_clique, replace=False)
        remaining_normal = set(str_normal_nodes_set) - set(feat_anomaly_nodes_set)
        index_set = [] # redundant
        for i in range(clique_size * num_clique):
            temp = np.random.choice(list(remaining_normal), k, replace=False)
            max_distance = 0
            index = 0
            for j in range(k):
                distance = np.linalg.norm(features[feat_anomaly_nodes_set[i], :] - features[temp[j], :])
                if distance > max_distance:
                    max_distance = distance
                    index = temp[j]
                index_set.append(index)  # redundant
                features[feat_anomaly_nodes_set[i], :] = features[index, :]

        final_anomaly = list(set(str_anomaly_nodes_set).union(set(feat_anomaly_nodes_set)))
        final_normal = list(remaining_normal)
        gnd = np.zeros(num_nodes)
        gnd[final_anomaly] = 1
        gnd_f = np.zeros(num_nodes)
        gnd_f[feat_anomaly_nodes_set] = 1
        gnd_s = np.zeros(num_nodes)
        gnd_s[str_anomaly_nodes_set] = 1
        return feat_anomaly_nodes_set, final_anomaly, final_normal, features, gnd, gnd_f, gnd_s
    else:
        raise Exception("data type not match")

    # str_anomaly_nodes_set, str_normal_nodes_set, adj = anomaly_injection_adj(adj, 20, 10)


# final_anomaly, final_normal, features, gnd = anomaly_injection_features(features, str_anomaly_nodes_set, str_normal_nodes_set, 20, 10, 10)


def make_ad_dataset_both_anomaly(dataset, clique_size, num_clique, k):
    if dataset == "citeseer" or dataset == "cora":
        adj, features = load_data(dataset)
    else:
        adj, features = load_other_data(dataset)
    str_anomaly_nodes_set, str_normal_nodes_set, adj = anomaly_injection_adj(adj, clique_size, num_clique)
    feat_anomaly_nodes_set, final_anomaly, final_normal, features, gnd, gnd_f, gnd_s = anomaly_injection_features(features, str_anomaly_nodes_set,
                                                                            str_normal_nodes_set, clique_size,
                                                                            num_clique, k)
    return adj, features, gnd, gnd_f, gnd_s





def make_ad_dataset_structure_anomaly(dataset, clique_size, num_clique, k):
    if dataset == "citeseer" or dataset == "cora":
        adj, features = load_data(dataset)
    else:
        adj, features = load_other_data(dataset)
    str_anomaly_nodes_set, str_normal_nodes_set, adj = anomaly_injection_adj(adj, clique_size, num_clique)
    gnd = np.zeros(features.shape[0])
    gnd[str_anomaly_nodes_set] = 1
    return adj, features, gnd


def make_ad_dataset_feature_anomaly(dataset, clique_size, num_clique, k):
    if dataset == "citeseer" or dataset == "cora":
        adj, features = load_data(dataset)
    else:
        adj, features = load_other_data(dataset)

    if isinstance(features, np.ndarray) is True:
        num_nodes = features.shape[0]
        feat_anomaly_nodes_set = np.random.choice(list(np.arange(num_nodes)), clique_size * num_clique, replace=False)
        remaining_normal = set(list(np.arange(num_nodes))) - set(feat_anomaly_nodes_set)
        for i in range(clique_size * num_clique):
            temp = np.random.choice(list(remaining_normal), k, replace=False)
            max_distance = 0
            index = 0
            for j in range(k):
                distance = np.linalg.norm(features[feat_anomaly_nodes_set[i], :] - features[temp[j], :])
                if distance > max_distance:
                    max_distance = distance
                    index = temp[j]
                features[feat_anomaly_nodes_set[i], :] = features[index, :]

        final_anomaly = list(set(feat_anomaly_nodes_set))
        final_normal = list(remaining_normal)
        gnd = np.zeros(num_nodes)
        gnd[final_anomaly] = 1
        return adj, features, gnd
    else:
        raise Exception("data type not match")




def pred_anomaly(error, clique_size, num_clique, mode):
    if mode == 0:
        num_anomaly = clique_size * num_clique * 2
    elif mode == 1:
        num_anomaly = clique_size * num_clique
    num_nodes = error.shape[0]
    pred_gnd = np.zeros(num_nodes)
    sort_index = np.argsort(error)
    pred_gnd[sort_index[num_nodes - num_anomaly:]] = 1
    return pred_gnd


def precision(pred_gnd, gnd):
    num_nodes = gnd.shape[0]
    count = 0
    for i in range(num_nodes):
        if pred_gnd[i] == gnd[i]:
            count = count + 1
    accuracy = count / num_nodes
    return accuracy


def load_ad_data(dataset_address):
    data = sio.loadmat(dataset_address)
    features = data["X"]
    features = torch.FloatTensor(features)
    adj = data["A"]

    if isinstance(adj, np.ndarray) is True:
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
    else:
        adj = nx.adjacency_matrix(nx.from_numpy_array(np.array(adj.todense())))
        # print(adj)
    gnd = torch.Tensor(data["gnd"])
    return adj, features, gnd


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def dim_reduction(A, pca=True, num_of_components=128):
    if not pca:
        num_of_components = A.shape[1]
    pca = PCA(n_components=num_of_components)
    A_pca = pca.fit_transform(A)
    scaler = StandardScaler()
    for i in range(np.shape(A_pca)[0]):
        A_pca[i, :] = scaler.fit_transform(A_pca[i, :].reshape(-1, 1)).reshape(-1)
    return A_pca



"""
        print("checking feature_decoder_layer_2 ")
        print(np.any(np.isnan(feature_decoder_layer_2.detach().numpy())))
        print(np.all(np.isfinite(feature_decoder_layer_2.detach().numpy())))

        print("checking structure_decoder_layer_2 ")
        print(np.any(np.isnan(structure_decoder_layer_2.detach().numpy())))
        print(np.all(np.isfinite(structure_decoder_layer_2.detach().numpy())))
"""
