import numpy as np

def load_all_pred_feature(file):
    pred_feature = np.loadtxt(file)
    return pred_feature

def load_all_query_train_feature(file):
    query_feature = np.loadtxt(file)
    return query_feature

def load_all_Oxford_query_test_feature(file):
    query_feature = []
    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            query_feature.append(cols)
    # print(pred_label)
    return query_feature

def load_all_Paris_query_test_feature(file):
    query_feature = []
    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            query_feature.append(cols[1:])
    # print(pred_label)
    return query_feature

def load_all_pred_label(file):
    pred_label = []
    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            label = cols[0]
            pred_label.append(label)
    # print(pred_label)
    return pred_label

def load_all_query_label(file):
    query_label = []
    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            label = cols[0]
            query_label.append(label)
    #print(query_label)
    return query_label

def load_all_pred_rank(file):
    pred_rank = []
    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            rank = cols[0]
            pred_rank.append(rank)
    #print(pred_rank)
    return pred_rank