import json
from sklearn.metrics import f1_score
import numpy as np
import copy
import fire

strg_list = ["question", "restatement or paraphrasing", "reflection of feelings", "self-disclosure", "affirmation and reassurance", 
             "providing suggestions", "information", "others"]

# preference modeling
def make_bt_table(data):
    bt_list=[]
    strategy_order={}
    for k, s in enumerate(strg_list):
        strategy_order[s]=k
        
    for s in strg_list:
        strg_num_list=[0]*8
        for d in data[s]:
            strg_num_list[strategy_order[d]]+=1
        bt_list.append(strg_num_list)
    bt_table = np.array(bt_list)
    return bt_table.T  

def estimate(bt_table, p_i):
    p = copy.deepcopy(p_i)
    for i in range(len(p_i)):
        model_p=0; human_p=0
        for j in range(len(p_i)):
            if j!=i:
                model_p += bt_table[i][j]*(p[j]/(p[i]+p[j]))
                human_p += bt_table[j][i]*(1/(p[i]+p[j]))
        p[i]=model_p/human_p    # update
    tmp=1; none_zero=0
    for p_h in p:
        if p_h>0:
            tmp*=p_h
            none_zero+=1
    for i in range(8):
        p[i]/=tmp**(1/none_zero)   # normalized by dividing by their geometric mean
        
    return p
            
def strg_preference(data, n=20):
    bt_table = make_bt_table(data)
    p = [1.0]*8   # initialize
    for _ in range(n):
        p = estimate(bt_table, p)
    
    total_p = sum(p)
    for i in range(len(p)):
        p[i]*=8     # To ensure that the sum of p equals 8."
        p[i]/= total_p
    
    return p

def strg_evaluation(data):
    strategy_order={} 
    strg_data={}    # strategy data gold(key), pred(value) 
    pred_data={}    # num of llm's predicted strategy of each strategy
    for k, s in enumerate(strg_list):
        strategy_order[s]=k
        pred_data[s]=0
        strg_data[s]=[]

    metrics={}  # total metric
    strg_metric = {}    # metric of each strategy

    gths=[]
    preds=[]
    for idx, item in enumerate(data):
        # total data
        gth=item['strg_gold'].lower()
        pred=item['strg_pred'].lower()
        gths.append(strategy_order[gth])
        
        if pred not in strategy_order.keys():
            print(pred, idx)
            pred = "others"
            
        preds.append(strategy_order[pred])
        
        # each strg data
        pred_strg=item['strg_pred'].lower()
        if pred_strg in strg_list:
            pred_data[pred_strg]+=1
            strg_data[item['strg_gold'].lower()].append(pred_strg)
        else:
            pred_data["others"]+=1
            strg_data[item['strg_gold'].lower()].append("others")
    weighted_f1 = f1_score(gths, preds, average='weighted')
    macro_f1 = f1_score(gths, preds, average='macro')
    strg_preferences = strg_preference(strg_data)  # Bradley–Terry Modeling
    
    # Metric of each strategy
    each_strg_f1_score = f1_score(gths, preds, average=None)
    strg_preference_list=[]; strg_f1=[]
    for strg in strg_list:
        strg_metrics={}
        
        strg_metrics["preference"] =  strg_preferences[strategy_order[strg]] 
        strg_preference_list.append(strg_preferences[strategy_order[strg]])
        
        strg_metrics['each f1 score'] = each_strg_f1_score[strategy_order[strg]]
        strg_f1.append(each_strg_f1_score[strategy_order[strg]])
        strg_metrics = {k: float(v) for k, v in strg_metrics.items()}
        strg_metric[strg] = strg_metrics
    
    # Total results print
    metrics['p_bias'] = np.std(strg_preference_list)
    metrics['weighted_f1_score'] = weighted_f1
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # Print Results
    name_strg_list = list(strg_metric.keys())
    print("Results of each strategy")
    print("Strategy | Preference | weighted-F1")
    for name in name_strg_list:
        m = strg_metric[name]
        y = f"{name}: {strg_metric[name]['preference']}, {strg_metric[name]['each f1 score']}"
        print(y)
    print()
    
    print(f"Total Results")
    print(f"Preference_bias : {metrics['p_bias']}")
    print(f"weighted-F1 : {metrics['weighted_f1_score']}")
    print(f"Macro F1 (Q) : {macro_f1}")
    
def checking_error(file_name, mode=None):
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    error_data=[]; total_data=[]
    for i, d in enumerate(data):
        if mode=="with_gold":
            d['strg_pred'] = d['strg_gold']
            total_data.append(d)
        elif mode=="random":
            d['strg_pred'] = d['strg_random']
            total_data.append(d)
        else:
            if d['strg_pred'].lower() in strg_list:
                total_data.append(d)
            else:
                error_data.append(d)
                print(f"Error Sample Id : {i}")
    
    return total_data, error_data

def evaluate_strategy(file_name):
    total_data, error_data = checking_error(file_name)
    
    if len(error_data)==0:
        strg_evaluation(total_data)
    else:
        print("Parsing error")

if __name__=="__main__":
    fire.Fire(evaluate_strategy)

# python metrics.py --file_name=results.json