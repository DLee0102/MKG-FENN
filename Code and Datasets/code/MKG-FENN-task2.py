import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
from modeltask2 import FusionLayer,GNN1,GNN2,GNN3,GNN4
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import warnings

import debugUtiles as db
import sys

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,choices=[100,500,1000,2000],default=100)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=5*1e-3) #4*1e-3

parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=65)

parser.add_argument("--n_drug",type=int,default=572)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=256)
args = parser.parse_args()

# seed = 0

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


'''
输入: 药物id字典, 数据集编号
输出: 知识图谱数据集, 尾节点长度, 关系长度
'''
def read_dataset(drug_name_id,num):
    # kg: key: drug_id, value: list of (tail_id, relation_id)
    kg = defaultdict(list)
    # tails: key: tail_name, value: tail_id
    tails = {}
    # relations: key: relation_name, value: relation_id 
    relations = {}
    
    drug_list=[]
    filename = "../dataset/dataset"+str(num)+".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split('//',2)
            head=string[0]
            tail=string[1]
            relation=string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
                
            # 对于DDI Matrix数据集需要做对称处理，这里这样处理同时还考虑了tail打标签要和head保持一致，因此没有用tails[tail]
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    
    # db.log(kg)
    return kg,len(tails),len(relations)

def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    # 将 机制 和 作用行为 合并 为 事件
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    # 统计每个 事件 出现的次数，输出为字典，key为 事件，value为次数，并从大到小排序
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    # list1 like [('mechanism action', 10), ('mechanism action', 8), ('mechanism action', 5)]
    
    # 得到每项不同事件的标签
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    
    # 将所有事件数据转换为标签
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])

    # 返回事件标签和事件类别的个数
    # db.log('envent num: '+str(len(count)))
    # db.log(new_label)
    return new_label,len(count)

def l2_re(parameter):
    reg=0
    for param in parameter:
        reg+=0.5*(param**2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes = np.arange(event_num))
    pred_one_hot = label_binarize(pred_type,classes = np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = 0.0
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = 0.0
    result_all[5] = 0.0
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = 0.0
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = 0.0
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]

def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task2'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def train(train_x,train_y,test_x,test_y,net,test_adj):
    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    # 对换两列的位置, like:[[drug_Aid, drug_Bid], [drug_Aid', drug_Bid'], ...]
    train_x[:,[0,1]] = train_x[:,[1,0]]
    # 为了保持对称性？？
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))

    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,65),dtype=torch.float)
    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        for x, y in train_iter:
            opti.zero_grad()
            train_acc = 0
            train_label = torch.LongTensor(y)
            x = torch.LongTensor(x)
            f_input = list()
            f_input.append(x)
            f_input.append(0)
            f_input.append(defaultdict(list))
            output = net(f_input)
            l = loss_function(output, train_label)
            l.backward()
            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output,dim=1), train_label)
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_x = torch.LongTensor(test_x)
            f_input = list()
            f_input.append(test_x)
            f_input.append(1)
            f_input.append(test_adj)
            test_output = F.softmax(net(f_input),dim=1)
            test_label = torch.LongTensor(test_y)
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            test_score = f1_score(torch.argmax(test_output,dim=1), test_label,average='macro')
            test_acc = accuracy_score(torch.argmax(test_output,dim=1), test_label)
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a),"test_score:",test_score)
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output



def find_dif(raw_matrix):
    sim_matrix4 = np.zeros((572, 572), dtype=float)
    for i in range(572):
        for j in range(572):
            for k in range(20):
                if i==j:
                    sim_matrix4[i,j] = 0
                    break
                else:
                    if raw_matrix[i,k] == raw_matrix[j,k]:
                        sim_matrix4[i, j] += 1
                    else:
                        sim_matrix4[i, j] += 0
    return sim_matrix4

def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator

def main():
    conn = sqlite3.connect("../dataset/event.db")
    
    # 药物数据
    df_drug = pd.read_sql('select * from drug;', conn)
    
    # 药物相互作用数据
    extraction = pd.read_sql('select * from extraction;', conn)
    
    # 提取药物相互作用数据的机制、作用行为、药物A、药物B
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    
    # 得到interaction数据标签和interaction类别的个数
    # newlabel: 所有interaction数据标签
    new_label,event_num = prepare(mechanism, action)
    new_label = np.array(new_label)

    # db.log('shape of new_label: '+str(new_label.shape))
    
    # 将每类药物的name转换为id，dict1是一个药物-id字典
    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)
    
    # db.log(dict1)
    
    # 将所有药物、药物A、药物B转换为id标签；dataset4和3刚好和论文里的part3和part4相反
    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1,1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1,2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1,4)
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    
    # x_datasets like: [[drugA_id1, drugB_id1], [drugA_id2, drugB_id2], ...]
    x_datasets = x_datasets.to_numpy()
    # db.log(x_datasets[0][0])
    
    # dataset_kg1 like: {drug_id1: [(tail_id1, relation_id1), (tail_id2, relation_id2), ...], ...}
    dataset={}
    dataset["dataset1"],dataset["dataset2"],dataset["dataset3"],dataset["dataset4"] = dataset1_kg,dataset2_kg,dataset3_kg,dataset4_kg
    tail_len={}
    tail_len["dataset1"],tail_len["dataset2"],tail_len["dataset3"],tail_len["dataset4"] = dataset1_tail_len,dataset2_tail_len,dataset3_tail_len,dataset4_tail_len
    relation_len={}
    relation_len["dataset1"],relation_len["dataset2"],relation_len["dataset3"],relation_len["dataset4"] = dataset1_relation_len,dataset2_relation_len,dataset3_relation_len,dataset4_relation_len
    train_sum, test_sum = 0, 0

    temp_kg = [defaultdict(list) for i in range(4)]
    for p,kg in enumerate(dataset):
        for i in dataset[kg].keys():
            for j in dataset[kg][i]:
                temp_kg[p][i].append(j[0])
    # temp_kg[0] like: [{drug_id1: [tail_id1, tail_id2, ...], ...}, ...]
    # db.log(temp_kg[0])
    
    # 572 is here db.log('drug_num: '+str(len(dict1)))
    
    # feature_matrix3 = np.zeros((572, 572), dtype=float)
    feature_matrix1 = np.zeros((572, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((572, dataset2_tail_len), dtype=float)
    feature_matrix3 = np.zeros((572, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((572, 572), dtype=float)
    
    db.log('feature_matrix1: '+str(feature_matrix1.shape))
    db.log('feature_matrix2: '+str(feature_matrix2.shape))  
    db.log('feature_matrix3: '+str(feature_matrix3.shape))
    db.log('feature_matrix4: '+str(feature_matrix4.shape))

    # db.log(dataset4_kg)
    for i in dataset4_kg.keys():
        for p,v in dataset4_kg[i]:
            # i: head, p: tail, v: relation; 分子结构矩阵
            feature_matrix3[i][p] = v
            
    # 每行代表一个head对应和哪些tail连同
    for i in temp_kg[0].keys():
        for j in temp_kg[0][i]:
            feature_matrix1[i][j] = 1

    for i in temp_kg[1].keys():
        for j in temp_kg[1][i]:
            feature_matrix2[i][j] = 1

    for i in temp_kg[2].keys():
        for j in temp_kg[2][i]:
            feature_matrix4[i][j] = 1

    # 计算各行之间的相似度，并得到相似度矩阵（对称矩阵）
    drug_sim1 = Jaccard(feature_matrix1)
    # feature_matrix2 = np.mat(feature_matrix2)
    drug_sim2 = Jaccard(feature_matrix2)
    # feature_matrix4 = np.mat(feature_matrix4)
    drug_sim4 = Jaccard(feature_matrix4)

    drug_sim3 = find_dif(feature_matrix3)

    # 统计每类interaction中药物A和药物B的数量
    # temp_drugA like: [[drugA_id1, drugA_id2, ...], ...], where the 0 dim is the interaction class
    temp_drugA = [[] for i in range(event_num)]
    temp_drugB = [[] for i in range(event_num)]
    for i in range(len(new_label)):
        temp_drugA[new_label[i]].append(drugA_id[i])
        temp_drugB[new_label[i]].append(drugB_id[i])

    # for i in range(event_num):
    #     if len(temp_drugA[i]) != len(temp_drugB[i]):
    #         db.log('error: drugA num != drugB num')
        # db.log('event '+str(i)+' drugA num: '+str(len(temp_drugA[i]))+' drugB num: '+str(len(temp_drugB[i])))
    # db.log(temp_drugA)
    
    # 作用：将药物类别集合分为5个子类，用于5折交叉验证
    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 5
            drug_cro_dict[temp_drugB[i][j]] = j % 5
    
    # db.log('drug_cro_dict: '+str(drug_cro_dict))
    
    train_drug = [[] for i in range(5)]
    test_drug = [[] for i in range(5)]
    # 将全部数据分为test data 和 train data,这种划分方式很奇怪
    for i in range(5):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    # db.log('4/5 drug_name: '+str(len(drug_name) * 0.8))
    # db.log('train_drug: '+str(len(train_drug[0])))
    
    # test_adj: （4：4个KG，5：5个交叉验证集），有4维
    test_adj = [[defaultdict(list) for _ in range(4)] for _ in range(5)]
    for i in range(5):
        for k in range(4):
            for j in test_drug[i]:
                if k == 0:
                    # 得到测试药物j的药物相似度向量，like: [[0.1, 0.2, 0.3, ...]]
                    target_list = drug_sim1[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list[0]):
                        # p是遍历所有药物的id，v是j关于p的相似度
                        # 如果这一相似度很大，并且p不在测试集中，且p不是当前测试药物，那么添加p到列表中
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)

                    # 第i折的第k个数据集的测试集的第j个药物对应的位置添加current_p，current_p的最后一项是相似度最大的训练集中的药物
                    # test_adj[i][k]是一个字典
                    test_adj[i][k][j].append(current_p)
                if k == 1:
                    target_list = drug_sim2[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                if k == 2:
                    target_list = drug_sim3[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                if k == 3:
                    target_list = drug_sim4[j].tolist()
                    max_v = 0
                    current_p = []
                    for p, v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
    y_true = np.array([])
    y_score = np.zeros((0, 65), dtype=float)
    y_pred = np.array([])
    
    for cross_ver in range(5):
        # Task 2 没有用到DDI Matrix数据集，即没用GNN4
        net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN2(dataset, tail_len, relation_len, args, dict1,drug_name),
                            GNN3(dataset, tail_len, relation_len, args, dict1,drug_name),
                            FusionLayer(args))
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(len(drugA)):
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_train.append(i)
                y_train.append(i)
            if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)
            # Task2 任务定义
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)

        # x_datasets是DDIML数据集中的所有行
        train_x = x_datasets[X_train]
        train_y = new_label[y_train]
        test_x = x_datasets[X_test]
        test_y = new_label[y_test]
        
        # 这种划分方式对于数据集的划分也不是均匀的
        db.log("Shape of train_x: "+str(train_x.shape))     # 25547
        db.log("Shape of test_x: "+str(test_x.shape))     # 10619

        # 传入的是test_adj对应的某一折的数据
        test_loss, test_acc, train_loss, train_acc, test_list, test_output = train(train_x, train_y, test_x, test_y,
                                                                                   net,test_adj[cross_ver])
        train_sum += train_acc
        test_sum += test_acc
        pred_type = torch.argmax(test_output, dim=1).numpy()
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, test_output))
        y_true = np.hstack((y_true, test_y))
        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
                cross_ver, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result("../result/", "all", result_all)
    save_result("../result/", "each", result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    main()