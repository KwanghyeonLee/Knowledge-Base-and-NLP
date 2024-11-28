import json
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import pandas as pd
import csv
import os
import time
import multiprocessing as mp
from itertools import product
import json
from tqdm import tqdm
import pandas as pd
import csv
import os
import ast

def read_json(filename):
    f = open(f'{filename}', 'r',encoding='utf-8-sig')
    data = []
    for line in f.readlines():
        dic = json.loads(line)
        data.append(dic)
    return data


def add_punct(str):
    str = str+'.'
    return str


def save_csv(content,path):
    with open(path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(content)

def remove_all(lst, element):
    return [x for x in lst if x != element]
    

def compare_sentences(org, org_data_i, org_data_j, exp_data_i, exp_data_j, speaker_id, row_index, row_name_i, inputs, persona_pair, sessions, episode_index, speaker, relation, source, tokenizer, i, j):

    org_data_i = remove_all(org_data_i, '')
    org_data_j = remove_all(org_data_j, '')

    org_data_i = remove_all(org_data_i, '\n')
    org_data_j = remove_all(org_data_j, '\n')

    org_data_i = remove_all(org_data_i, '.')
    org_data_j = remove_all(org_data_j, '.')

    if not isinstance(org_data_i, list) or not isinstance(org_data_j, list) or not isinstance(exp_data_i, list) or not isinstance(exp_data_j, list):
        print(org)
        print(org_data_i, org_data_j, exp_data_i, exp_data_j)
        import pdb; pdb.set_trace()

    for orig_sentence1, orig_sentence2 in product(org_data_i, org_data_j):
        if orig_sentence1 == 'none' or orig_sentence2 == 'none':
            continue
        if isinstance(orig_sentence1, list) or isinstance(orig_sentence2, list) or len(orig_sentence1) < 2 or len(orig_sentence2) < 2:
            print(org)
            print(orig_sentence1, orig_sentence2, speaker_id, row_name_i, i)
            print('parsing error')
            import pdb; pdb.set_trace()
        inputs.append(f"{orig_sentence1} {tokenizer.sep_token} {orig_sentence2}")
        persona_pair.append([orig_sentence1, orig_sentence2])
        sessions.append([f'session{i}', f'session{j}'])
        episode_index.append(row_name_i)
        speaker.append(speaker_id)
        relation.append('inter_session')
        source.append([orig_sentence1, orig_sentence2])

    for orig_sentence1, exp_sentence2 in product(org_data_i, exp_data_j):
        if orig_sentence1 == 'none' or exp_sentence2 == 'none':
            continue
        if isinstance(orig_sentence1, list) or isinstance(exp_sentence2, list) or len(orig_sentence1) < 2 or len(exp_sentence2) < 2:
            print(org)
            print(orig_sentence1, exp_sentence2, speaker_id, row_name_i, i)
            print('parsing error')
            import pdb; pdb.set_trace()
        inputs.append(f"{orig_sentence1} {tokenizer.sep_token} {exp_sentence2}")
        persona_pair.append([orig_sentence1, exp_sentence2])
        sessions.append([f'session{i}', f'session{j}'])
        episode_index.append(row_name_i)
        speaker.append(speaker_id)
        relation.append('inter_session')
        source.append([orig_sentence1, orig_sentence2])
    
    for exp_sentence1, orig_sentence2 in product(exp_data_i, org_data_j):
        if exp_sentence1 == 'none' or orig_sentence2 == 'none':
            continue
        if isinstance(exp_sentence1, list) or isinstance(orig_sentence2, list) or len(exp_sentence1) < 2 or len(orig_sentence2) < 2:
            print(org)
            print(exp_sentence1, orig_sentence2, speaker_id, row_name_i, i)
            print('parsing error')
            import pdb; pdb.set_trace()
        inputs.append(f"{exp_sentence1} {tokenizer.sep_token} {orig_sentence2}")
        persona_pair.append([exp_sentence1, orig_sentence2])
        sessions.append([f'session{i}', f'session{j}'])
        episode_index.append(row_name_i)
        speaker.append(speaker_id)
        relation.append('inter_session')
        source.append([orig_sentence1, orig_sentence2])

    for exp_sentence1, exp_sentence2 in product(exp_data_i, exp_data_j):
        if exp_sentence1 == 'none' or exp_sentence2 == 'none':
            continue
        if isinstance(exp_sentence1, list) or isinstance(exp_sentence2, list) or len(exp_sentence1) < 2 or len(exp_sentence2) < 2:
            print(org)
            print(exp_sentence1, exp_sentence2, speaker_id, row_name_i, i)
            print('parsing error')
            import pdb; pdb.set_trace()
        inputs.append(f"{exp_sentence1} {tokenizer.sep_token} {exp_sentence2}")
        persona_pair.append([exp_sentence1, exp_sentence2])
        sessions.append([f'session{i}', f'session{j}'])
        episode_index.append(row_name_i)
        speaker.append(speaker_id)
        relation.append('inter_session')
        source.append([orig_sentence1, orig_sentence2])
                    
    return inputs, persona_pair, sessions, episode_index, speaker, relation, source


def minimize_list(nested_list):
    while isinstance(nested_list, list) and len(nested_list) == 1:
        nested_list = nested_list[0]
    if len(nested_list) == 2:
        nested_list = [nested_list]
    return nested_list


def flatten(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


def single_check(lst):
    if len(lst) == 2 and isinstance(lst[1], list) and not isinstance(lst[0], list):
        return lst[1]
    else:
        return minimize_list(lst)


def model_process(model, all_splits, return_dict, idx):
    batch_splits, persona_splits, sessions_splits, episode_index_splits, speaker_splits, relation_splits, source_splits = all_splits
    outputs, personas, sessions, episodes, speakers, relations, sources = [], [], [], [], [], [], []
    
    for input, persona, session, episode, speaker, relation, source in tqdm(zip(batch_splits, persona_splits, sessions_splits, episode_index_splits, speaker_splits, relation_splits, source_splits)): 
        output = model(input)
        outputs += output   
        personas += persona
        sessions += session
        episodes += episode
        speakers += speaker
        relations += relation
        sources += source
    
    return_dict[idx] = [outputs, personas, sessions, episodes, speakers, relations, sources]


def parallel_inference(models, chunks):
    manager = mp.Manager()
    return_dict = manager.dict()

    batch_chunks, persona_chunks, sessions_chunks, episode_index_chunks, speaker_chunks, relation_chunks, source_chunks = chunks
    
    processes = []
    idx = 0
    for model, batch, persona, sessions, episode_index, speaker, relation, source in zip(models, batch_chunks, persona_chunks, sessions_chunks, episode_index_chunks, speaker_chunks, relation_chunks, source_chunks):
        chunk = [batch, persona, sessions, episode_index, speaker, relation, source]
        p = mp.Process(target=model_process, args=(model, chunk, return_dict, idx))
        p.start()
        processes.append(p)
        time.sleep(1)
        idx += 1

    for p in processes:
        p.join()
        
    return return_dict


def divide_list(lst, div):
    N = len(lst)
    base_length = N // div
    remainder = N % div

    parts = []
    start_idx = 0
    for i in range(div):
        if i < remainder:
            end_idx = start_idx + base_length + 1
        else:
            end_idx = start_idx + base_length
        parts.append(lst[start_idx:end_idx])
        start_idx = end_idx

    return parts


def comet_intersession_dnli_prediction(models, tokenizer, gpu_idx, input_expanded_filtered_filepath, UPDATED_PKB, session, args=None, persona_type="comet"):
    session_num = int(''.join(filter(str.isdigit, session)))
    i = session_num
    j = i + 1

    df_next = pd.read_csv(input_expanded_filtered_filepath, index_col=0)

    inputs, persona_pair, sessions, episode_index, speaker, relation, source = [], [], [], [], [], [], []

    for row_index, row_name_i in enumerate(UPDATED_PKB.keys()):
        for speaker_and_session in UPDATED_PKB[row_name_i].keys():
            if 'A' in speaker_and_session:
                
                if df_next.loc[row_name_i, f'comet_A_session{j}'] == '"[[]]"':
                    pass
                else:
                    inputs, persona_pair, sessions, episode_index, speaker, relation, source = compare_sentences(
                        ast.literal_eval(df_next.loc[row_name_i, f'comet_A_session{j}']),
                        flatten([k for k in UPDATED_PKB[row_name_i][speaker_and_session].keys()]),
                        flatten([item[0] if isinstance(item[0], list) else [item[0]] for item in single_check(ast.literal_eval(df_next.loc[row_name_i, f'comet_A_session{j}']))]),
                        flatten([v for v in UPDATED_PKB[row_name_i][speaker_and_session].values()]),
                        flatten([item[1] if isinstance(item[1], list) else [item[1]] for item in single_check(ast.literal_eval(df_next.loc[row_name_i, f'comet_A_session{j}']))]),
                        1,
                        row_index, row_name_i,
                        inputs, persona_pair, sessions, episode_index, speaker, relation, source,
                        tokenizer,
                        int(speaker_and_session[-1]),
                        j
                    )        
            elif 'B' in speaker_and_session:
                if df_next.loc[row_name_i, f'comet_B_session{j}'] == '"[[]]"':
                    pass
                else:
                    inputs, persona_pair, sessions, episode_index, speaker, relation, source = compare_sentences(
                        ast.literal_eval(df_next.loc[row_name_i, f'comet_B_session{j}']),
                        flatten([k for k in UPDATED_PKB[row_name_i][speaker_and_session].keys()]),
                        flatten([item[0] if isinstance(item[0], list) else [item[0]] for item in single_check(ast.literal_eval(df_next.loc[row_name_i, f'comet_B_session{j}']))]),
                        flatten([v for v in UPDATED_PKB[row_name_i][speaker_and_session].values()]),
                        flatten([item[1] if isinstance(item[1], list) else [item[1]] for item in single_check(ast.literal_eval(df_next.loc[row_name_i, f'comet_B_session{j}'])) ]),
                        2,
                        row_index, row_name_i, 
                        inputs, persona_pair, sessions, episode_index, speaker, relation, source,
                        tokenizer,
                        int(speaker_and_session[-1]), 
                        j
                    )

    # Making predictions
    batch_splits = [inputs[k:k+args.batch_size] for k in range(0, len(inputs), args.batch_size)]
    persona_splits = [persona_pair[k:k+args.batch_size] for k in range(0, len(persona_pair), args.batch_size)]
    sessions_splits = [sessions[k:k+args.batch_size] for k in range(0, len(sessions), args.batch_size)]
    episode_index_splits = [episode_index[k:k+args.batch_size] for k in range(0, len(episode_index), args.batch_size)]
    speaker_splits = [speaker[k:k+args.batch_size] for k in range(0, len(speaker), args.batch_size)]
    relation_splits = [relation[k:k+args.batch_size] for k in range(0, len(relation), args.batch_size)]
    source_splits = [source[k:k+args.batch_size] for k in range(0, len(source), args.batch_size)]

    batch_chunks = divide_list(batch_splits, len(gpu_idx))
    persona_chunks = divide_list(persona_splits, len(gpu_idx))
    sessions_chunks = divide_list(sessions_splits, len(gpu_idx))
    episode_index_chunks = divide_list(episode_index_splits, len(gpu_idx))
    speaker_chunks = divide_list(speaker_splits, len(gpu_idx))
    relation_chunks = divide_list(relation_splits, len(gpu_idx))
    source_chunks = divide_list(source_splits, len(gpu_idx))

    all_chunks = [batch_chunks, persona_chunks, sessions_chunks, episode_index_chunks, speaker_chunks, relation_chunks, source_chunks]

    result_tmp = parallel_inference(models, all_chunks)

    content = []
    for j in range(len(gpu_idx)):
        outputs, personas, sessions, episodes, speakers, relations, sources = result_tmp[j]
        for k in range(len(outputs)):
            episode_dic = {
                'persona_pair': personas[k],
                'predicted_label': outputs[k],
                'sessions': sessions[k],
                'episode_index': episodes[k],
                'speaker': speakers[k],
                'relation': relations[k],
                'source_persona': sources[k]
            }
            content.append(episode_dic)

    return content


def convert_json_to_csv(json_data):
    csv_data = [["episode", "session", "sentence", "options"]]

    for episode_idx, sessions in json_data.items():
        for session, sentences in sessions.items():
            for sentence, options in sentences.items():
                csv_data.append([episode_idx, session, sentence, options])

    return csv_data
