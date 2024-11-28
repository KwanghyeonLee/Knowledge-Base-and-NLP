import json
from tqdm import tqdm
import pandas as pd
import csv
import os
import time
import multiprocessing as mp
from itertools import combinations

class PersonaData:
    def __init__(self, args):
        self.data = {}
        self.args = args

    def load_comet_persona(self, cnt):
        df = pd.read_csv(self.args.input_path, index_col=0)
        session_a, session_b = {}, {}
        for row_idx, row_name in enumerate(df.index):
            if row_idx == cnt:
                break
            sentences_a = eval(df[f'comet_A_session{self.args.session_num}'][row_name])
            sentences_b = eval(df[f'comet_B_session{self.args.session_num}'][row_name])

            session_a[row_name] = [sentences_a]
            session_b[row_name] = [sentences_b]

        self.data[f'session{self.args.session_num}'] = {
            'comet_A_session': session_a,
            'comet_B_session': session_b
        }

    def get_session_data(self, session_idx, persona_type):
        return self.data[f'session{session_idx}'][f'{persona_type}_A_session'], self.data[f'session{session_idx}'][f'{persona_type}_B_session']


def process_sentences(data, row_name, tokenizer, speaker_id, relation_type, session_idx):
    inputs, persona_pair, sessions, episode_index, speaker, relation, source_persona  = [], [], [], [], [], [], []
    
    if data == '[[]]':
        pass
    else:
        for idx in range(len(data)):
            orig_persona = data[idx][0] 
            exp_personas = data[idx][1] 

            for i, exp_persona1 in enumerate(exp_personas):
                for j, exp_persona2 in enumerate(exp_personas):
                    if i >= j or isinstance(exp_persona1, list) or isinstance(exp_persona2, list):
                        continue
                    if exp_persona1.strip() == 'none' or exp_persona2.strip() == 'none':
                        continue
                    inputs.append(f"{exp_persona1} {tokenizer.sep_token} {exp_persona2}")
                    persona_pair.append([exp_persona1, exp_persona2])
                    sessions.append(f'session{session_idx}')
                    episode_index.append([row_name])
                    speaker.append(speaker_id)
                    relation.append(relation_type)
                    source_persona.append([orig_persona,orig_persona])

    return inputs, persona_pair, sessions, episode_index, speaker, relation, source_persona


def compare_personas(row_name, session_datas, tokenizer, speaker_id, relation_type, session_idx):
    inputs, persona_pair, sessions, episode_index, speaker, relation, source_persona = [], [], [], [], [], [], []

    session_data = session_datas[0]

    if session_data == '[[]]':
        pass
    else:
        for persona_data1, persona_data2 in combinations(session_data, 2):
            try:
                orig_persona1, exp_personas1 = persona_data1
                orig_persona2, exp_personas2 = persona_data2
            except:
                import pdb; pdb.set_trace()
                
            if orig_persona1.strip() != 'none' and orig_persona2.strip() != 'none':
                if f"{orig_persona2} {tokenizer.sep_token} {orig_persona1}" in inputs or f"{orig_persona1} {tokenizer.sep_token} {orig_persona2}" in inputs:
                    continue
                inputs.append(f"{orig_persona1} {tokenizer.sep_token} {orig_persona2}")
                persona_pair.append([orig_persona1, orig_persona2])
                sessions.append(f'session{session_idx}')
                episode_index.append([row_name])
                speaker.append(speaker_id)
                relation.append(relation_type)
                source_persona.append([orig_persona1, orig_persona2])

            # Compare orig1 with each exp of persona2
            for exp_persona in exp_personas2:
                if orig_persona1.strip() != 'none' and exp_persona.strip() != 'none':
                    if f"{exp_persona} {tokenizer.sep_token} {orig_persona1}" in inputs or f"{orig_persona1} {tokenizer.sep_token} {exp_persona}" in inputs:
                        continue
                    inputs.append(f"{orig_persona1} {tokenizer.sep_token} {exp_persona}")
                    persona_pair.append([orig_persona1, exp_persona])
                    sessions.append(f'session{session_idx}')
                    episode_index.append([row_name])
                    speaker.append(speaker_id)
                    relation.append(relation_type)
                    source_persona.append([orig_persona1, orig_persona2])

            # Compare each exp of persona1 with orig2
            for exp_persona in exp_personas1:
                if exp_persona.strip() != 'none' and orig_persona2.strip() != 'none': 
                    if f"{orig_persona2} {tokenizer.sep_token} {exp_persona}" in inputs or f"{exp_persona} {tokenizer.sep_token} {orig_persona2}" in inputs:
                        continue
                    inputs.append(f"{exp_persona} {tokenizer.sep_token} {orig_persona2}")
                    persona_pair.append([exp_persona, orig_persona2])
                    sessions.append(f'session{session_idx}')
                    episode_index.append([row_name])
                    speaker.append(speaker_id)
                    relation.append(relation_type)
                    source_persona.append([orig_persona1, orig_persona2])

            # Compare each pair of exp personas from persona1 and persona2
            for exp_persona1 in exp_personas1:
                for exp_persona2 in exp_personas2:
                    if exp_persona1.strip() == 'none' or exp_persona2.strip() == 'none':
                        continue
                    if f"{exp_persona2} {tokenizer.sep_token} {exp_persona1}" in inputs or f"{exp_persona1} {tokenizer.sep_token} {exp_persona2}" in inputs:
                        continue
                    inputs.append(f"{exp_persona1} {tokenizer.sep_token} {exp_persona2}")
                    persona_pair.append([exp_persona1, exp_persona2])
                    sessions.append(f'session{session_idx}')
                    episode_index.append([row_name])
                    speaker.append(speaker_id)
                    relation.append(relation_type)
                    source_persona.append([orig_persona1, orig_persona2])

    return inputs, persona_pair, sessions, episode_index, speaker, relation, source_persona


def process_session_data(persona_data, session_idx, tokenizer, speaker_id, relation_type):
    inputs, persona_pair, sessions, episode_index, speaker, relation, source_persona = [], [], [], [], [], [], []
    
    for row_name in tqdm(persona_data.keys()):
        session_data = persona_data[row_name]  # Fetching all persona data for the current session
        if relation_type == 'intra':
            for persona1_idx, data in enumerate(session_data): 
                # persona_expansion_1 = data  # Single persona expansion data
                results = process_sentences(data, row_name, tokenizer, speaker_id, relation_type, session_idx)
                inputs.extend(results[0])
                persona_pair.extend(results[1])
                sessions.extend(results[2])
                episode_index.extend(results[3])
                speaker.extend(results[4])
                relation.extend(results[5])
                source_persona.extend(results[6])
                
        elif relation_type == 'inter':
            results = compare_personas(row_name, session_data, tokenizer, speaker_id, relation_type, session_idx)
            inputs.extend(results[0])
            persona_pair.extend(results[1])
            sessions.extend(results[2])
            episode_index.extend(results[3])
            speaker.extend(results[4])
            relation.extend(results[5])
            source_persona.extend(results[6])
            
            
    return inputs, persona_pair, sessions, episode_index, speaker, relation, source_persona


def comet_dnli_prediction(args, tokenizer, session_idx, batch_size, persona_type="comet", relation_type ='intra'):

    cnt = args.episode_num
    
    if  relation_type == 'intra': 
        data_loader = PersonaData(args)
        data_loader.load_comet_persona(cnt)
        persona_data_A, persona_data_B = data_loader.get_session_data(session_idx, persona_type)
    
        print(f"*** Predicting Session{session_idx} intra consistency ***")
        inputs_A, persona_pair_A, sessions_A, episode_index_A, speaker_A, relation_A, src_persona_A = process_session_data(persona_data_A, session_idx, tokenizer, 1, 'intra')
        inputs_B, persona_pair_B, sessions_B, episode_index_B, speaker_B, relation_B, src_persona_B = process_session_data(persona_data_B, session_idx, tokenizer, 2, 'intra')
        
        inputs_A.extend(inputs_B)
        persona_pair_A.extend(persona_pair_B)
        sessions_A.extend(sessions_B)
        episode_index_A.extend(episode_index_B)
        speaker_A.extend(speaker_B)
        relation_A.extend(relation_B)
        src_persona_A.extend(src_persona_B)

    elif relation_type == 'inter':
        data_loader = PersonaData(args)
        data_loader.load_comet_persona(cnt)
        persona_data_A, persona_data_B = data_loader.get_session_data(session_idx, persona_type)

        print(f"*** Predicting Session{session_idx} inter consistency ***")
        inputs_A, persona_pair_A, sessions_A, episode_index_A, speaker_A, relation_A, src_persona_A = process_session_data(persona_data_A, session_idx, tokenizer, 1, 'inter')
        inputs_B, persona_pair_B, sessions_B, episode_index_B, speaker_B, relation_B, src_persona_B = process_session_data(persona_data_B, session_idx, tokenizer, 2, 'inter')
 
        inputs_A.extend(inputs_B)
        persona_pair_A.extend(persona_pair_B)
        sessions_A.extend(sessions_B)
        episode_index_A.extend(episode_index_B)
        speaker_A.extend(speaker_B)
        relation_A.extend(relation_B)
        src_persona_A.extend(src_persona_B)
    
    batch_splits = [inputs_A[i:i+batch_size] for i in range(0, len(inputs_A), batch_size)]
    persona_splits = [persona_pair_A[i:i+batch_size] for i in range(0, len(persona_pair_A), batch_size)]
    sessions_splits = [sessions_A[i:i+batch_size] for i in range(0, len(sessions_A), batch_size)]
    episode_index_splits = [episode_index_A[i:i+batch_size] for i in range(0, len(episode_index_A), batch_size)]
    speaker_splits = [speaker_A[i:i+batch_size] for i in range(0, len(speaker_A), batch_size)]
    relation_splits = [relation_A[i:i+batch_size] for i in range(0, len(relation_A), batch_size)]
    src_persona_splits = [src_persona_A[i:i+batch_size] for i in range(0, len(src_persona_A), batch_size)]
    
    return batch_splits, persona_splits, sessions_splits, episode_index_splits, speaker_splits, relation_splits, src_persona_splits


def model_process(model, all_splits, return_dict, idx):
    batch_splits, persona_splits, sessions_splits, episode_index_splits, speaker_splits, relation_splits, src_persona_splits = all_splits
    outputs, personas, sessions, episodes, speakers, relations, src_personas = [], [], [], [], [], [], []
    
    for input, persona, session, episode, speaker, relation, src_persona in tqdm(zip(batch_splits, persona_splits, sessions_splits, episode_index_splits, speaker_splits, relation_splits, src_persona_splits)): 
        output = model(input)
        outputs += output   
        personas += persona
        sessions += session
        episodes += episode
        speakers += speaker
        relations += relation
        src_personas += src_persona
    
    return_dict[idx] = [outputs, personas, sessions, episodes, speakers, relations, src_personas]


def parallel_inference(models, chunks):
    manager = mp.Manager()
    return_dict = manager.dict()

    batch_chunks, persona_chunks, sessions_chunks, episode_index_chunks, speaker_chunks, relation_chunks, src_chunks = chunks
    
    processes = []
    idx = 0
    for model, batch, persona, sessions, episode_index, speaker, relation, src_persona in zip(models, batch_chunks, persona_chunks, sessions_chunks, episode_index_chunks, speaker_chunks, relation_chunks, src_chunks):
        chunk = [batch, persona, sessions, episode_index, speaker, relation, src_persona]
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


def save_csv(content, path):
    with open(path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(content)


def main_nli_intra(models, tokenizer, gpu_idx, args):
    
    persona_type = args.expansion_type
    batch_size = args.batch_size

    session_num = args.session_num        

    content = []

    for consistency in ['intra', 'inter']:

        batch_splits, persona_splits, sessions_splits, episode_index_splits, speaker_splits, relation_splits, src_persona_splits = comet_dnli_prediction(args, tokenizer, session_num, batch_size, persona_type=persona_type, relation_type=consistency)
        batch_chunks = divide_list(batch_splits, len(gpu_idx))
        persona_chunks = divide_list(persona_splits, len(gpu_idx))
        sessions_chunks = divide_list(sessions_splits, len(gpu_idx))
        episode_index_chunks = divide_list(episode_index_splits, len(gpu_idx))
        speaker_chunks = divide_list(speaker_splits, len(gpu_idx))
        relation_chunks = divide_list(relation_splits, len(gpu_idx))
        src_personas  = divide_list(src_persona_splits, len(gpu_idx))

        all_chunks = [batch_chunks, persona_chunks, sessions_chunks, episode_index_chunks, speaker_chunks, relation_chunks, src_personas]
    
        result_tmp = parallel_inference(models, all_chunks)
        
        results = []
        for j in range(len(gpu_idx)):
            outputs, personas, sessions, episodes, speakers, relations, src_personas = result_tmp[j]
            for k in range(len(outputs)):
                obs_dict = {
                    'persona_pair': personas[k],
                    'predicted_label': outputs[k],
                    'sessions': sessions[k],
                    'episode_index': episodes[k][0],
                    'speaker': speakers[k],
                    'consistency_relation': relations[k],
                    'source_persona': src_personas[k]
                }
                results.append(obs_dict)
                
        print(f"Number of results collected ({consistency}): {len(results)}") 
        
        content += results
    
    path = args.output_path
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)
    print(f"file saved on {path}")

class get_args:
    def __init__(self, expansion_type='comet', batch_size=(1024*4), num_gpu=4):

        self.expansion_type = expansion_type
        self.batch_size = batch_size
        self.num_gpu = num_gpu
