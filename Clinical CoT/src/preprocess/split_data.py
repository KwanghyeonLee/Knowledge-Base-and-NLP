import math
import json
import random
import re
import numpy as np
from shutil import copyfile
import os

def Unimodal_split_data(get_type, input_dir, output_dir, percentage):
    input_json = os.path.join(input_dir, f"{get_type}.json")
    with open(input_json, 'r') as f:
        load_json = json.load(f)['data']
    print(f"total_length: {len(load_json)}")
    total_D = 0
    total_NC = 0
    total_MCI = 0

    for data in load_json:
        label = data['org_label']
        if label == 'Normal Cognition':
            total_NC +=1
        elif label == 'Dementia':
            total_D +=1
        elif label == 'Mild Cognitive Impairment':
            total_MCI +=1
    extract_D = int(round(percentage*total_D))
    extract_NC = int(round(percentage*total_NC))
    extract_MCI = int(round(percentage*total_MCI))
    finish_flag = False
    result_list = []
    D_counter = 0
    NC_counter = 0
    MCI_counter = 0
    picked_list = []
    while not finish_flag:
        if NC_counter == extract_NC and D_counter == extract_D and MCI_counter == extract_MCI:
            finish_flag = True
            break
        find_new_random_index = False
        while not find_new_random_index:
            random_index = random.randrange(1,len(load_json))
            if random_index not in picked_list:
                picked_list.append(random_index)
                find_new_random_index = True
                get_random_data = load_json[random_index]
                get_label = get_random_data['org_label']
                break
        if get_label == 'Normal Cognition':
            if NC_counter == extract_NC:
                continue
            else:
                result_list.append(get_random_data)
                NC_counter +=1
        elif get_label == 'Dementia':
            if D_counter == extract_D:
                continue
            else:
                result_list.append(get_random_data)
                D_counter +=1
        elif get_label == 'Mild Cognitive Impairment':
            if MCI_counter == extract_MCI:
                continue
            else:
                result_list.append(get_random_data)
                MCI_counter +=1
    result_dict = {}
    result_dict['data'] = result_list
    save_dir = save_dir = os.path.join(output_dir, str(int(percentage*100)))
    save_file = os.path.join(output_dir, str(int(percentage*100)),f'{get_type}.json')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_file, 'w') as wf:
        json.dump(result_dict, wf, indent=4, ensure_ascii=False)


def multimodal_split_data(get_type, total_input_dir, multimodal_dir, output_dir, percentage):
    input_json = os.path.join(total_input_dir, f"{get_type}.json")
    with open(input_json, 'r') as f:
        load_json = json.load(f)['data']
    print(f"json length: {len(load_json)}")
    total_D = 0
    total_NC = 0
    total_MCI = 0

    for data in load_json:
        label = data['org_label']
        if label == 'Normal Cognition':
            total_NC +=1
        elif label == 'Dementia':
            total_D +=1
        elif label == 'Mild Cognitive Impairment':
            total_MCI +=1
    extract_D = int(round(percentage*total_D))
    extract_NC = int(round(percentage*total_NC))
    extract_MCI = int(round(percentage*total_MCI))
    data_list = sorted(os.listdir(multimodal_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    finish_flag = False
    result_list = []
    D_counter = 0
    NC_counter = 0
    MCI_counter = 0
    picked_list = []
    while not finish_flag:
        if NC_counter == extract_NC and D_counter == extract_D and MCI_counter == extract_MCI:
            finish_flag = True
            break
        find_new_random_index = False
        while not find_new_random_index:
            random_index = random.sample(data_list, 1)[0]
            if random_index not in picked_list:
                picked_list.append(random_index)
                find_new_random_index = True
                get_random_index = random_index
                get_path = os.path.join(multimodal_dir, get_random_index)
                data = np.load(get_path,allow_pickle=True)
                regex = re.compile('\(([ABC]+)')
                extract_text = regex.findall(str(data['label']))
                answer = extract_text[0]
                break
        if answer == 'C':
            if NC_counter == extract_NC:
                continue
            else:
                result_list.append(get_random_index)
                NC_counter +=1
        elif answer == 'B':
            if MCI_counter == extract_MCI:
                continue
            else:
                result_list.append(get_random_index)
                MCI_counter +=1        
        elif answer == 'A':
            if D_counter == extract_D:
                continue
            else:
                result_list.append(get_random_index)
                D_counter +=1 
    assert D_counter == extract_D and NC_counter == extract_NC and MCI_counter == extract_MCI            
    for i in result_list:
        get_path = os.path.join(multimodal_dir, i)
        go_dir = os.path.join(output_dir, str(int(percentage*100)))
        if not os.path.exists(go_dir):
            os.makedirs(go_dir)
        go_path = os.path.join(go_dir, i)
        copyfile(get_path,go_path) 