import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score
import random
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from multiprocessing.pool import Pool # 进程池


def get_metrics(y_true, y_pred, add=""):
    y_pred_int = (y_pred > 0.5).astype(np.int32)
    acc = metrics.accuracy_score(y_true, y_pred_int)
    auc = metrics.roc_auc_score(y_true, y_pred)
    return {f"{add}acc": acc, f"{add}auc": auc}

def load_pred_result(model_list,dataset_list,model_root_dir):
    df_dict = {}
    report_list = []
    for dataset in dataset_list:
        for model_name in tqdm(model_list):
            # 获取模型路径
            model_dir = os.path.join(model_root_dir, dataset, model_name)
            if not os.path.exists(model_dir):  # 部分模型没有相关数据集
                print(f"skip {dataset} {model_name}")
            else:
                auc_list = []
                df_len_list = []
                for fold_name in os.listdir(model_dir):
                    if "report.csv" in fold_name or "report_100.csv" in fold_name or "report_50.csv" in fold_name:
                        continue
                    elif ".ipynb_checkpoints" in fold_name:
                        continue
                    fold_dir = os.path.join(model_dir, fold_name)
                    config = json.load(open(os.path.join(fold_dir, "config.json")))
                    fold = config['params']['fold']
                    #                 if fold!=0:
                    #                     continue
                    df = pd.read_csv(
                        os.path.join(fold_dir, "late_mean_win_new.csv"))
                    df_len_list.append(len(df))
                    # add cols
                    if "late_trues" in df.columns:
                        df['y_true'] = df['late_trues']
                    if "late_mean" in df.columns:
                        df['y_pred'] = df['late_mean']
                    auc = round(roc_auc_score(df['y_true'], df['y_pred']), 4)
                    auc_list.append(auc)
                    df_dict[f'{dataset}_{model_name}_{fold}'] = df
                if len(set(df_len_list)) != 1:
                    print(f"{dataset} {model_name} len is not name {df_len_list}")
                report_list.append({
                    "dataset": dataset,
                    "model_name": model_name,
                    "auc": round(np.mean(auc_list), 4),
                    "std": round(np.std(auc_list), 4)
                })
    return df_dict,report_list

def sample_dataset(model_list,dataset_list,df_dict,num_repeat=10,num_sample=10000):
    new_dataset = {}
    report_list = []
    for dataset in dataset_list:
        for model_name in model_list:
            df = df_dict[f'{dataset}_{model_name}_0']#默认用第一折的结果
            auc_list = []
            for repeat_index in range(num_repeat):
                df_sample = df.sample(n=num_sample, random_state=repeat_index).copy()
                new_dataset[f'{dataset}_{model_name}_{repeat_index}'] = df_sample
                auc = round(roc_auc_score(df_sample['y_true'], df_sample['y_pred']), 4)
                auc_list.append(auc)
            report_list.append({
                "dataset": dataset,
                "model_name": model_name,
                "auc": round(np.mean(auc_list), 4),
                "std": round(np.std(auc_list), 4)
            })
    return new_dataset,report_list
    

def allocation_humans(condition_model, model_pred, model_name, dataset, repeat_index, human_type_config, human_dict, config, debug=False):
    """分配专家，不限制人工成本
    
    Args:
        condition_model (_type_): _description_
        model_pred (_type_): _description_
        dataset (_type_): _description_
        human_type_config (_type_): _description_
        human_dict (_type_): _description_
        config (_type_): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    roles = np.where(condition_model, 0, 1)  # true -> 0 , false -> 1
    condition_human = ~condition_model  # 数组元素取反 TRUE -> FALSE
    human_num = roles.sum()  # human的数量
    num_interaction = len(model_pred)
    human_pred = np.zeros(num_interaction)
    cost = 0
    
    # keep_index获取打乱的human索引
    keep_index = np.arange(0, len(condition_model))[condition_human]
    if len(keep_index)==0:
        return human_pred, human_num, roles, cost
    
    if debug:
        print(f"human_num is {human_num}")
    if config['mode'] == "S":
        human_pred =  human_dict[f"{dataset}_{model_name}_{repeat_index}_S"]
        cost = human_type_config["S"]['human_cost']*human_num
    elif config['mode'] == "A":
        human_pred =  human_dict[f"{dataset}_{model_name}_{repeat_index}_A"]
        cost = human_type_config["A"]['human_cost']*human_num
    
    # human平均分配
    elif config['mode'] == "average":
        if debug:
            print(f"len keep_index is {len(keep_index)}")
        np.random.seed(1)
        np.random.shuffle(keep_index)
        # 将index三等分并存入list
        #split_index = [keep_index[0:int(1/3*human_num)],  keep_index[int(1/3*human_num):int(2/3*human_num)],  keep_index[int(2/3*human_num):]]
        split_index = [ keep_index[0:int(1/2*human_num)], keep_index[int(1/2*human_num):] ]
        #print(split_index)
        # 开始赋值
        human_pred[split_index[0]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_S"][split_index[0]]
        human_pred[split_index[1]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_A"][split_index[1]]
        #human_pred[split_index[2]] = human_dict[f"{dataset}_B"][split_index[2]]

        num_list = [np.sum(condition_human[split_index[x]]) for x in range(2)]
        cost = 0
        for num, human_type in zip(num_list, ['S', 'A']):
            one_cost = num * human_type_config[human_type]['human_cost']
            cost += one_cost
        if debug:
            print(f"human_num is {human_num}, np.sum(num_list) is {np.sum(num_list)}, num_list is {num_list}")
        assert human_num == np.sum(num_list)
    
    # 根据成本权重分配
    elif config['mode'] == "conf_average":
        if debug:
            print(f"len keep_index is {len(keep_index)}")
        
        model_confi = np.abs(model_pred - 0.5) / 0.5
        sort_index = model_confi[keep_index].argsort()  # 从小到大排序 返回索引下标
        keep_index = keep_index[sort_index]
        
        split_index = [ keep_index[0:int(1/2*human_num)], keep_index[int(1/2*human_num):] ]
        
        if debug:
            print( f"s_ratio is {s_ratio},a_ratio is {a_ratio},len is {[len(x) for x in split_index]}")
        # 开始赋值
        human_pred[split_index[0]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_S"][split_index[0]]
        human_pred[split_index[1]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_A"][split_index[1]]
        #human_pred[split_index[2]] = human_dict[f"{dataset}_B"][split_index[2]]

        num_list = [np.sum(condition_human[split_index[x]])
                    for x in range(2)]
        cost = 0
        for num, human_type in zip(num_list, ['S', 'A']):
            one_cost = num * human_type_config[human_type]['human_cost']
            cost += one_cost
        if debug:
            print(
            f"human_num is {human_num},np.sum(num_list) is {np.sum(num_list)},num_list is {num_list}")
        
        assert human_num == np.sum(num_list)
    
    # 按照置信度排序后设置阈值分配
    elif config['mode'] == "conf_threshold":
        
        model_confi = np.abs(model_pred - 0.5) / 0.5
        #sort_index = model_confi[keep_index].argsort()  # 从小到大排序 返回索引下标
        sort_index1 = np.where( model_confi<0.5 )
        sort_index2 = np.where( model_confi>=0.5 )
        s_index = np.intersect1d(keep_index, sort_index1)
        a_index = np.intersect1d(keep_index, sort_index2)
        #keep_index = keep_index[sort_index]
        if debug:
            print(f"len keep_index is {len(keep_index)}")
        #total_cost = sum([1/x['human_cost'] for x in human_type_config.values()])
        
        split_index = [ s_index, a_index ]
        if debug:
            print(
            f"s_ratio is {s_ratio}, a_ratio is {a_ratio}, len is {[len(x) for x in split_index]}")
        # 开始赋值
        human_pred[split_index[0]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_S"][split_index[0]]
        human_pred[split_index[1]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_A"][split_index[1]]

        num_list = [ np.sum(condition_human[split_index[x]]) for x in range(2) ]
        cost = 0
        for num, human_type in zip(num_list, ['S', 'A']):
            one_cost = num * human_type_config[human_type]['human_cost']
            cost += one_cost
        if debug:
            print(f"human_num is {human_num},np.sum(num_list) is {np.sum(num_list)},num_list is {num_list}")
        #print(f"human_num:{human_num}, np.sum(num_list):{np.sum(num_list)}")
        assert human_num == np.sum(num_list)
    
    # 按照置信度排序后 成本权重分配
    elif config['mode'] == "conf_weight_average":
    
        if debug:
            print(f"len keep_index is {len(keep_index)}")
        total_cost = sum([1/x['human_cost']
                            for x in human_type_config.values()])
        s_ratio = (1/human_type_config['S']['human_cost'])/total_cost
        a_ratio = (1/human_type_config['A']['human_cost'])/total_cost
        #b_ratio = (1/human_type_config['B']['human_cost'])/total_cost
        
        model_confi = np.abs(model_pred - 0.5) / 0.5
        sort_index = model_confi[keep_index].argsort()  # 从小到大排序 返回索引下标
        keep_index = keep_index[sort_index]
        
        split_index = [ keep_index[0:int(s_ratio*human_num)], keep_index[int((s_ratio)*human_num):] ]
        if debug:
            print(
            f"s_ratio is {s_ratio},a_ratio is {a_ratio},len is {[len(x) for x in split_index]}")
        # 开始赋值
        human_pred[split_index[0]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_S"][split_index[0]]
        human_pred[split_index[1]
                    ] = human_dict[f"{dataset}_{model_name}_{repeat_index}_A"][split_index[1]]

        num_list = [np.sum(condition_human[split_index[x]])
                    for x in range(2)]
        cost = 0
        for num, human_type in zip(num_list, ['S', 'A']):
            one_cost = num * human_type_config[human_type]['human_cost']
            cost += one_cost
        if debug:
            print(
            f"human_num is {human_num},np.sum(num_list) is {np.sum(num_list)},num_list is {num_list}")
        assert human_num == np.sum(num_list)

    return human_pred, human_num, roles, cost






dataset_map = {
    "assist2009": "AS2009",
    "algebra2005": "Algebra2005",
    "bridge2algebra2006": "BD2006",
    "nips_task34": "NIPS34",
    "statics2011": "Statics2011",
    "assist2015": "ASSISTments2015",
    "poj": "POJ",
    "peiyou":"XES3G5M",
    "ednet":"EdNet"
}
modelname_map = {
    "dkt": "DKT",
    "dkt+": "DKT+",
    "dkt_forget": "DKT-F",
    "kqn": "KQN",
    "dkvmn": "DKVMN",
    "atktfix": "ATKTFIX", 
    "gkt": "GKT",
    "sakt": "SAKT",
    "saint": "SAINT",
    "akt": "AKT",
    "skvmn": "SKVMN",
    "hawkes": "HAWKES",
    "deep_irt": "DeepIRT",
    "lpkt": "LPKT",
    "iekt": "IEKT",
    "qdkt": "qDKT",
    "cdkt": "AT-DKT",
    "bakt": "simpleKT",
    "bakt_time": "timeSKT",
    "qikt": "QIKT"
}
col_map = {"auc":"AUC","acc":"Accuracy","f1":"F1"}
col_map.update(dataset_map)
col_map.update(modelname_map)
