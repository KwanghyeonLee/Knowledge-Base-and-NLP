
from typing import List, Tuple
import json
from tqdm import tqdm
import copy
import pandas as pd
import logging
import ast
import pickle
from collections import defaultdict
import time
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from transformers import AutoTokenizer, pipeline

from nli_model_inference_intra_inter import main_nli_intra, get_args
from nli_model_inference_intersession import comet_intersession_dnli_prediction
from utils import get_least_used_gpu, get_least_used_gpu_multiple
from persona_assesment_graph import *
from argument import load_parser_and_args

def flatten_to_individual_strings(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            for sub_item in item:
                result.append(str(sub_item))
        else:
            result.append(str(item))
    return result


def convert_format_for_refine(evidences, speaker, use_source):
    result = []
    for evidence in evidences:
        try:
            dialogue_fragments = ' '.join(evidence.get('dialogue_fragment', []) if isinstance(evidence.get('dialogue_fragment', []), list) else [evidence.get('dialogue_fragment', [])])
        except:
            evidence['dialogue_fragment'] = flatten_to_individual_strings(evidence['dialogue_fragment'])
            dialogue_fragments = ' '.join(evidence.get('dialogue_fragment', []) if isinstance(evidence.get('dialogue_fragment', []), list) else [evidence.get('dialogue_fragment', [])])

        result.append(f'Dialogue fragments of Persona {speaker}: {dialogue_fragments}')
        if use_source:
            source_persona = evidence.get('source_persona', '')
            result.append(f'Source Persona: {source_persona}')
        else:
            pass
    return result

def is_nested_empty_list(lst):
    if not isinstance(lst, list):
        return False
    if not lst:
        return True
    return all(is_nested_empty_list(item) for item in lst)

def self_logging(text):
    print(text)
    with open(f'runs/{args.save_memo}_cost_time_record.txt', 'a', encoding='utf-8') as f:
        f.write(f"{text}\n")

class ConflictPersonaGraph:
    def __init__(self):
        self.graph = {}
        self.PKB = {}
        
    def __str__(self):
        return f"Graph: {json.dumps(self.graph, indent=4)}\nPKB: {json.dumps(self.PKB, indent=4)}\nDialog Tree: {json.dumps(self.dialog_tree, indent=4)}"
    
    def load_edges_from_file(self, data: str):
        for entry in data:
            node1, node2 = entry["persona_pair"]
            
            weight = entry["predicted_label"]["score"]
            speaker = entry["speaker"]
            episode_index = entry["episode_index"]

            if isinstance(entry["sessions"], list):
                session_i, session_j = entry["sessions"]
            else:
                session_i, session_j = entry["sessions"], entry["sessions"]

            if isinstance(episode_index, list):
                raise TypeError(f"episode_index cannot be a list. Current value: {episode_index}")
            
            self.add_edge(node1, node2, weight, speaker, episode_index, session_i, session_j)
    
    def test_persona_in_dialog_tree(self, data):
        self_logging("Test if all personas are in dialog_tree.")
        for entry in tqdm(data):
            for i, persona in enumerate(entry["persona_pair"]):
                if isinstance(entry["sessions"], list):
                    self.recall_conversation(persona, entry["episode_index"], entry["speaker"], entry["sessions"][i])
                else:
                    self.recall_conversation(persona, entry["episode_index"], entry["speaker"], entry["sessions"])
        self_logging("Test passed.")
    
    def add_edge(self, node1: str, node2: str, weight: float, speaker: int, episode_index: str, session_i, session_j):
        if episode_index not in self.graph:
            self.graph[episode_index] = {1: {}, 2: {}}
            
        if episode_index not in self.PKB:
            self.PKB[episode_index] = {f"comet_A_{session}": {}, f"comet_B_{session}": {}}
        
        if (session_i, node1) not in self.graph[episode_index][speaker]:
            self.graph[episode_index][speaker][(session_i, node1)] = {}
        self.graph[episode_index][speaker][(session_i, node1)][(session_j, node2)] = weight

        if (session_j, node2) not in self.graph[episode_index][speaker]:
            self.graph[episode_index][speaker][(session_j, node2)] = {}
        self.graph[episode_index][speaker][(session_j, node2)][(session_i, node1)] = weight
        
    def process_dialog_data(self, src2dialog, exp2src, args):
        # Load persona_to_text
        with open(src2dialog, 'rb') as file:
            self.persona_to_text = pickle.load(file)

        # Load expanded_persona_to_dialog
        with open(exp2src, 'rb') as file:
            self.expanded_persona_to_dialog = pickle.load(file)
    
    def merge_and_verify_dictionaries(self, dict1):        
        dict2 = self.PKB
        
        def merge_values(value1, value2):
            """Merge two values which can be either lists or dictionaries."""
            if isinstance(value1, list) and isinstance(value2, list):
                return list(set(value1 + value2))  # Merge lists and remove duplicates
            
            elif isinstance(value1, dict) and isinstance(value2, dict):
                # Merge dictionaries recursively
                for key in value2:
                    if key in value1:
                        value1[key] = merge_values(value1[key], value2[key])
                    else:
                        value1[key] = value2[key]
                return value1
            else:
                # If values are not mergeable, return the original value1
                return value1

    # Merge dictionaries recursively
        for test_key in dict1:
            if test_key not in dict2:
                dict2[test_key] = dict1[test_key]
            else:
                for session_key in dict1[test_key]:
                    if session_key not in dict2[test_key]:
                        dict2[test_key][session_key] = dict1[test_key][session_key]
                    else:
                        for key in dict1[test_key][session_key]:
                            if key in dict2[test_key][session_key]:
                                dict2[test_key][session_key][key] = merge_values(
                                    dict2[test_key][session_key][key], dict1[test_key][session_key][key])
                            else:
                                dict2[test_key][session_key][key] = dict1[test_key][session_key][key]
        
        # Verify the merged dictionary
        for test_key in dict1:
            assert test_key in dict2, f"Key '{test_key}' not found in merged dictionary"
            for session_key in dict1[test_key]:
                assert session_key in dict2[test_key], f"Key '{session_key}' not found in merged dictionary"
                for key in dict1[test_key][session_key]:
                    assert key in dict2[test_key][session_key], f"Key '{key}' not found in merged dictionary"
            
        return dict2

               
    def recall_conversation(self, persona: str, episode_index, speaker, session):
        results = []

        try: # if persona is from expanded one
            # expanded_persona_to_dialog -> exp2src
            dialogue_fragment = self.expanded_persona_to_dialog[(episode_index, str(speaker), session, persona)][0]

            # persona_to_text -> src2dialog
            source_persona = self.expanded_persona_to_dialog[(episode_index, str(speaker), session, persona)][1]
            
            results.append({
                "dialogue_fragment": dialogue_fragment,
                "source_persona": source_persona
            })

        except: # if persona is from source
            dialogue_fragment = self.persona_to_text[(episode_index, str(speaker), session, persona)][0]
            results.append({
                    "dialogue_fragment": dialogue_fragment,
                    "source_persona": persona
            })

        if not results:
            self_logging('Dialogue fragment no match issue')
            exit()
        
        return results
    
    def self_reflective_personality_assessment(self, node_1: str, node_2: str, evidence_1: List[str],evidence_2: List[str],args, episode_index, speaker, asses_idx, session, src_node1, src_node2) -> Tuple[bool, bool]:
        
        final_decision, cost = cot_reasoning(node_1, node_2, evidence_1, evidence_2, args, episode_index, speaker, asses_idx, session, src_node1, src_node2)

        return final_decision, cost
    
    def update_PKB(self, episode_index, session, node, speaker):        
        if episode_index not in self.PKB.keys():
            self.PKB[episode_index] = {f"comet_A_{session}": {}, f"comet_B_{session}": {}}
        
        speaker_and_session =  f"comet_{'A' if speaker == 1 else 'B'}_{session}"
        if speaker_and_session not in self.PKB[episode_index].keys():
            self.PKB[episode_index][speaker_and_session] = {}

        try: # if persona is from expanded one        
            src = self.expanded_persona_to_dialog[(episode_index, str(speaker), session, node)][1]
            if src not in self.PKB[episode_index][speaker_and_session]:
                self.PKB[episode_index][speaker_and_session][src] = [node]
            else:
                if node not in self.PKB[episode_index][speaker_and_session][src]:
                    self.PKB[episode_index][speaker_and_session][src].append(node)
                
        except: # if persona is from source 
            self.PKB[episode_index][speaker_and_session][node] = []
        
    def top_two_important_nodes(self, graph, episode_index, speaker):
        node_values = self.calculate_node_values(graph, episode_index, speaker)

        return node_values
    
    def update_graph(self, graph, nodes_to_keep, nodes_to_remove, episode_index=None, speaker=None):
        for node in nodes_to_remove:
            if node in graph.keys():
                del graph[node]
                for neighbor in graph:
                    graph[neighbor].pop(node, None)
            else:
                print(f"Node {node} is not in graph of {episode_index}, {speaker}")

        if len(nodes_to_keep) == 2:
            graph[nodes_to_keep[0]].pop(nodes_to_keep[1], None)
            graph[nodes_to_keep[1]].pop(nodes_to_keep[0], None)

        disconnected_nodes = [node for node, neighbors in graph.items() if not neighbors]
        
        for node in disconnected_nodes:
            del graph[node]
    
    def calculate_node_values(self, graph, episode_index, speaker):
        degrees = {node: len(neighbors) for node, neighbors in graph.items()}

        max_degree_sum = -1
        max_edge_sum = -1
        max_pair = None

        for node, neighbors in graph.items():
            for neighbor in neighbors:
                degree_sum = degrees[node] + degrees[neighbor] - 1
            
                node_edge_sum = sum(graph[node][n] for n in graph[node])
                neighbor_edge_sum = sum(graph[neighbor][n] for n in graph[neighbor])
                total_edge_sum = node_edge_sum + neighbor_edge_sum - graph[node][neighbor]

                if total_edge_sum > max_edge_sum or (total_edge_sum == max_edge_sum and degree_sum > max_degree_sum):    
                    max_degree_sum = degree_sum
                    max_edge_sum = total_edge_sum
                    max_pair = (node, neighbor)

        return max_pair

    def fast_chatgpt_multiprocessing(self, session, episode_index: str, speaker: int, args): 
        step_count = 0        
        original_count = 0
        graph = self.graph[episode_index][speaker]
        selected_node_pairs = []
        prev_selected_nodes = None
        result_dict = {}  
        episode_cost = 0
        load_record = 0
        gpt_call = 0

        while len(graph) > 1:
            step_count += 1
            sessions_nodes = self.top_two_important_nodes(graph, episode_index, speaker)
            sessions_nodes_1, sessions_nodes_2 = sessions_nodes
            session_i, node1 = sessions_nodes_1
            session_j, node2 = sessions_nodes_2
            
            if not sessions_nodes or len(sessions_nodes) < 2:
                continue
            
            if prev_selected_nodes == sessions_nodes:
                self.update_graph(graph, [], sessions_nodes, episode_index, speaker)
                continue
            
            prev_selected_nodes = copy.deepcopy(sessions_nodes)
            
            if sessions_nodes in selected_node_pairs or sessions_nodes[::-1] in selected_node_pairs:
                continue
            
            selected_node_pairs.append(sessions_nodes)

            try:
                src_node1 = self.expanded_persona_to_dialog[(episode_index, str(speaker), session_i, node1)][1]
            except:
                src_node1 = node1

            try:
                src_node2 = self.expanded_persona_to_dialog[(episode_index, str(speaker), session_j, node2)][1]
            except:
                src_node2 = node2

            if node1 == src_node1 and node2 != src_node2:
                self.update_graph(graph, [(session_j, node2)], [(session_i, node1)], episode_index, speaker) 
                original_count += 1

            elif node1 != src_node1 and node2 == src_node2:
                self.update_graph(graph, [(session_i, node1)], [(session_j, node2)], episode_index, speaker) 
                original_count += 1                

            else:
                org_evidence_1 = self.recall_conversation(node1, episode_index, speaker, session_i)
                org_evidence_2 = self.recall_conversation(node2, episode_index, speaker, session_j)
                
                converted_evidence_1 = convert_format_for_refine(org_evidence_1, 1, args.use_source)
                converted_evidence_2 = convert_format_for_refine(org_evidence_2, 2, args.use_source)
                
                try:
                    with open(os.path.join(args.assess_result_dir, args.save_memo + args.parsing_test, session, 'results_{}_{}_{}_{}.json'.format(args.model_type, session, speaker, episode_index)), 'r', encoding='utf-8') as f:
                        decision_file = json.load(f)
                    
                    if episode_index != decision_file[f'assesment_idx_{step_count}']['episode_index']:
                        raise ValueError('No match for episode_index')
                    
                    if speaker != decision_file[f'assesment_idx_{step_count}']['speaker']:
                        raise ValueError('No match for speaker')

                    asses_idx = decision_file[f'assesment_idx_{step_count}']['assesment_idx']
                    assert asses_idx == step_count, 'assess_idx != stepcount'

                    refined_personas = decision_file[f'assesment_idx_{step_count}']['refined_personas']
                    cost = decision_file[f'assesment_idx_{step_count}']['cost']

                    load_record += 1

                except Exception as e:
                    refined_personas, cost = self.self_reflective_personality_assessment(node1, node2, converted_evidence_1, converted_evidence_2, args, episode_index, speaker, step_count, session, src_node1, src_node2)
                    
                    gpt_call += 1
                if len(refined_personas) == 1:
                    self.update_graph(graph, [], sessions_nodes, episode_index, speaker)

                elif len(refined_personas) == 2:
                    self.update_graph(graph, [], sessions_nodes, episode_index, speaker)
                
                episode_cost += cost

        print(f"Done: {session}, speaker:{speaker}, {episode_index}, org_win_exp:{original_count}, load_record:{load_record}, gpt_call:{gpt_call}, cost: {episode_cost}", flush=True)
        with open(f'runs/{args.save_memo}.txt', 'a') as f:
            f.write(f"Done: {session}, speaker:{speaker}, {episode_index}, org_win_exp:{original_count}, load_record:{load_record}, gpt_call:{gpt_call}, cost: {episode_cost}\n")
        
        result_dict['persona_to_text'] = self.persona_to_text
        result_dict['cost'] = episode_cost

        if step_count == 0:
            result_dict['step_count_zero'] = 1
        else:
            result_dict['step_count_zero'] = 0

        return result_dict
    
    def node_removal_simulations(self, session, episode_index: str, speaker: int, args):
        step_count = 0        
        original_count = 0
        graph = self.graph[episode_index][speaker]
        selected_node_pairs = []
        prev_selected_nodes = None
        result_dict = {}
        episode_cost = 0
        load_record = 0
        gpt_call = 0

        while len(graph) > 1:
            step_count += 1
            sessions_nodes = self.top_two_important_nodes(graph, episode_index, speaker)
            sessions_nodes_1, sessions_nodes_2 = sessions_nodes
            session_i, node1 = sessions_nodes_1
            session_j, node2 = sessions_nodes_2
            
            if not sessions_nodes or len(sessions_nodes) < 2:
                continue
            
            if prev_selected_nodes == sessions_nodes:
                self.update_graph(graph, [], sessions_nodes, episode_index, speaker)
                continue
            
            prev_selected_nodes = copy.deepcopy(sessions_nodes)
            
            if sessions_nodes in selected_node_pairs or sessions_nodes[::-1] in selected_node_pairs:
                continue
            
            selected_node_pairs.append(sessions_nodes)

            try:
                src_node1 = self.expanded_persona_to_dialog[(episode_index, str(speaker), session_i, node1)][1]
            except:
                src_node1 = node1

            try:
                src_node2 = self.expanded_persona_to_dialog[(episode_index, str(speaker), session_j, node2)][1]
            except:
                src_node2 = node2

            if node1 == src_node1 and node2 != src_node2:
                self.update_graph(graph, [(session_j, node2)], [(session_i, node1)], episode_index, speaker) 
                self.update_PKB(episode_index, session_i, node1, speaker)
                original_count += 1
                
            elif node1 != src_node1 and node2 == src_node2:
                self.update_graph(graph, [(session_i, node1)], [(session_j, node2)], episode_index, speaker)
                self.update_PKB(episode_index, session_j, node2, speaker)
                original_count += 1
                
            else:
                org_evidence_1 = self.recall_conversation(node1, episode_index, speaker, session_i)
                org_evidence_2 = self.recall_conversation(node2, episode_index, speaker, session_j)
                
                converted_evidence_1 = convert_format_for_refine(org_evidence_1, 1, args.use_source)
                converted_evidence_2 = convert_format_for_refine(org_evidence_2, 2, args.use_source)
                
                try:
                    with open(os.path.join(args.assess_result_dir, args.save_memo + args.parsing_test, session, 'results_{}_{}_{}_{}.json'.format(args.model_type, session, speaker, episode_index)), 'r', encoding='utf-8') as f:
                        decision_file = json.load(f)
                    
                    if episode_index != decision_file[f'assesment_idx_{step_count}']['episode_index']:
                        raise ValueError('No match for episode_index')
                    
                    if speaker != decision_file[f'assesment_idx_{step_count}']['speaker']:
                        raise ValueError('No match for speaker')

                    asses_idx = decision_file[f'assesment_idx_{step_count}']['assesment_idx']
                    assert asses_idx == step_count, 'assess_idx != stepcount'

                    refined_personas = decision_file[f'assesment_idx_{step_count}']['refined_personas']
                    cost = decision_file[f'assesment_idx_{step_count}']['cost']

                    load_record += 1

                except Exception as e: 
                    refined_personas, cost = self.self_reflective_personality_assessment(node1, node2, converted_evidence_1, converted_evidence_2, args, episode_index, speaker, step_count, session, src_node1, src_node2)
                    
                    gpt_call += 1
                    
                dialog1 = org_evidence_1[0]['dialogue_fragment']
                dialog2 = org_evidence_2[0]['dialogue_fragment']
                
                dialogue_fragment = [dialog1, dialog2]
                
                if len(refined_personas) == 1:
                    self.update_graph(graph, [], sessions_nodes, episode_index, speaker)

                    self.persona_to_text[(episode_index, str(speaker), session, refined_personas[0])] = [dialogue_fragment]
                    
                    self.update_PKB(episode_index, session, refined_personas[0], speaker)

                elif len(refined_personas) == 2:
                    self.update_graph(graph, [], sessions_nodes, episode_index, speaker)

                    self.persona_to_text[(episode_index, str(speaker), session, refined_personas[0])] = [dialogue_fragment[0]]
                    self.persona_to_text[(episode_index, str(speaker), session, refined_personas[1])] = [dialogue_fragment[1]]

                    self.update_PKB(episode_index, session, refined_personas[0], speaker)
                    self.update_PKB(episode_index, session, refined_personas[1], speaker)    
                
                episode_cost += cost

        result_dict['persona_to_text'] = self.persona_to_text
        result_dict['cost'] = episode_cost

        if step_count == 0:
            result_dict['step_count_zero'] = 1
        else:
            result_dict['step_count_zero'] = 0

        return result_dict
    

def process_episode(episode_idx, input_expanded_filtered, input_conflict):
    episode_data = {}
    for speaker_idx in input_expanded_filtered.columns:
        episode_data[speaker_idx] = {}
        
        PKB_tmp = ast.literal_eval(input_expanded_filtered.loc[episode_idx, speaker_idx])
        if all(len(sublist) == 0 for sublist in PKB_tmp):
            continue

        filtered_by_episode = [item for item in copy.deepcopy(input_conflict) if item['episode_index'] == episode_idx]
        persona_pairs_values = [pair for item in filtered_by_episode for pair in item['persona_pair']]
        Conflict_tmp = set(persona_pairs_values)
        
        PKB_filtered = []
        if str(PKB_tmp) == '[[]]':
            pass
        else:
            for gold_persona_sent, expanded_persona_list in PKB_tmp:
                
                if gold_persona_sent not in Conflict_tmp:
                    new_expanded_persona_list = [phrase for phrase in expanded_persona_list if phrase not in Conflict_tmp]
                    if new_expanded_persona_list:
                        PKB_filtered.append([gold_persona_sent, new_expanded_persona_list])
                    
        for gp_ep in PKB_filtered:
            gp = gp_ep[0]
            ep = gp_ep[1]
            episode_data[speaker_idx][gp] = ep

    print(f'{episode_idx} is done.', end=' ', flush=True)

    return episode_idx, episode_data


def deep_update(target, source):
    """Recursively update a dictionary with another dictionary."""
    
    for key, value in source.items():
        if isinstance(value, dict):
            target[key] = deep_update(target.get(key, {}), value)
        else:
            if key in target.keys():
                target[key].extend(value)
            else:
                target[key] = value
    return target


if __name__ == "__main__":
    start_time = time.time()
    parser, args = load_parser_and_args()
    if args.mp_spawn:
        mp.set_start_method('spawn')
        mp_spawn = True
    else:
        mp_spawn = False

    logging.basicConfig(filename=f'{args.log_dir}/{args.save_memo}.log', filemode='w', format='%(message)s', level=logging.INFO)
    self_logging(args)
    self_logging("="*50)
    
    sessions = ["session1","session2", "session3", "session4"]

    intersession_conflict = None
    
    self_logging(f'*** Initiate ConflictPersonaGraph ***')
    conflict_persona_graph = ConflictPersonaGraph() 

    self_logging(f'*** Initiate dialogue tree ***')
    src2dialog = f'data/dialogue_tree_new/persona_to_text{args.epi_num}_{args.expansion}.pkl'
    exp2src = f'data/dialogue_tree_new/expanded_persona_to_dialog{args.epi_num}_{args.expansion}.pkl'
    conflict_persona_graph.process_dialog_data(src2dialog, exp2src, args)
    
    experiment_total_cost = 0
    model = None
    models = None

    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    
    for session in sessions:
        self_logging(f"*************** {session} start! ***************")
        
        session_num = int(''.join(filter(str.isdigit, session)))

        input_expanded_filepath = args.input_expanded_filepath_before + f"{session}" + args.input_expanded_filepath_after + '.csv'
        input_expanded_filtered_filepath = args.input_expanded_filepath_before + f"{session}" + args.input_expanded_filepath_after + f'_{args.save_memo}' + '_filtered.csv'
        
        input_conflict_filepath = f'data/{args.expansion}_nli_1001/{args.expansion}_{session}_intra_inter_persona_prediction_th_0{args.epi_num}_{args.save_memo}.json'
        
        input_conflict_intersession_filepath = f'output/pkb_intersession_conflict/{args.expansion}_session{session_num-1}_{session_num}_inter_session_persona_prediction_th_0{args.epi_num}_{args.save_memo}.json'
        
        input_conflict_total_filepath = f'output/input_conflict_total/input_conflict_total_{args.save_memo}_{session}{args.epi_num}.json'

        self_pkb_filepath = f'output/self_pkb/self_pkb_{args.save_memo}_{session}{args.epi_num}.json'

        output_PKB_filepath = f'output/final_pkb_updated{args.epi_num}/final_pkb_{session}{args.epi_num}_{args.save_memo}.json'
        
        input_expanded = pd.read_csv(input_expanded_filepath, index_col=0)
        
        if os.path.exists(input_expanded_filtered_filepath):
            input_expanded_filtered = pd.read_csv(input_expanded_filtered_filepath, index_col=0)
            self_logging(f"*** Load {session} input_expanded_filtered : {input_expanded_filtered_filepath} ***")
            
        else:
            if model is None:
                gpu_list = ast.literal_eval(args.gpu_list)
                selectec_gpu = get_least_used_gpu(gpu_list)
                
                model = pipeline("text-classification", model="roberta-large-mnli", device=selectec_gpu, batch_size=batch_size)
            
            persona_dict = {}

            for idx, row in input_expanded.iterrows():
                persona_dict[idx] = {}
                
                for column in [f'comet_A_session{session_num}', f'comet_B_session{session_num}']:
                    double_list = ast.literal_eval(row[column])
                    
                    persona_dict[idx][column] = {}
                    
                    if is_nested_empty_list(double_list):
                        persona_dict[idx][column] = [[]]
                    else:
                        for original, expanded_list in double_list:                  
                            persona_dict[idx][column][original] = expanded_list
                    
            persona_pairs = []
            persona_loc = {}

            cnt = 0
            for idx, data in persona_dict.items():
                for column, sub_data in data.items():
                    if isinstance(sub_data, list):
                        continue
                    for original, expanded_list in sub_data.items():
                        for expanded in expanded_list:
                            persona_loc[cnt] = (idx, column, original, expanded)
                            persona_pairs.append(f"{original} {tokenizer.sep_token} {expanded}")
                            cnt += 1

            self_logging(f"persona_pairs: {len(persona_pairs)}")

            batch_splits = [persona_pairs[i:i+batch_size] for i in range(0, len(persona_pairs), batch_size)]
            
            outputs = []
            for batch in batch_splits:
                outputs += model(batch)

            for i, dict_output in enumerate(outputs):
                dict_output['persona_pairs'] = persona_pairs[i]
                
            with open(f'{args.input_expanded_filepath_before}/ori_exp_pair_nli_{session}_{args.epi_num}_{args.save_memo}.json', 'w', encoding='utf-8') as f:
                json.dump(outputs, f, indent=4)

            removal_index = [idx for idx, item in enumerate(outputs) if item['label'] == 'CONTRADICTION']
            
            self_logging(f"removal_index: {len(removal_index)}")

            for remove in removal_index:
                idx, column, original, expanded = persona_loc[remove]
                
                rm_expand = persona_dict[idx][column][original].index(expanded)
                persona_dict[idx][column][original].pop(rm_expand)
                
            input_expanded_filtered = pd.DataFrame(columns=[f'comet_A_session{session_num}', f'comet_B_session{session_num}'])
            for idx, data in persona_dict.items():
                new_data = {}
                for column, persona_dicts in data.items():
                    if isinstance(persona_dicts, list):
                        new_data[column] = '"[[]]"'
                        self_logging(new_data[column])
                    else:
                        persona_lists = [[k,v] for k,v in persona_dicts.items()]
                        new_data[column] = str(persona_lists)
                
                input_expanded_filtered.loc[idx] = new_data
            
            input_expanded_filtered.to_csv(input_expanded_filtered_filepath, encoding='utf-8')
            self_logging(f"*** Save {session} input_expanded_filtered ***")

        if os.path.exists(input_conflict_filepath):
            with open(input_conflict_filepath, 'r', encoding='utf-8-sig') as f:
                input_conflict = json.load(f)
            self_logging(f"*** Load {session} input_conflict : {input_conflict_filepath} ***")
        else:
            nli_intra_args = get_args()
            nli_intra_args.input_path = input_expanded_filtered_filepath
            nli_intra_args.output_path = input_conflict_filepath
            nli_intra_args.episode_num = len(input_expanded_filtered.index)
            nli_intra_args.session_num = session_num
            if mp_spawn == False:
                mp_spawn = True
            
            if models is None:
                gpu_list = ast.literal_eval(args.gpu_list)
                multi_gpu_idx = get_least_used_gpu_multiple(gpu_list, args.num_gpu)

                if model is not None:
                    del model
                models = [pipeline("text-classification", model="roberta-large-mnli", device=i, batch_size=batch_size) for i in multi_gpu_idx]
            main_nli_intra(models, tokenizer, multi_gpu_idx, nli_intra_args)

            with open(input_conflict_filepath, 'r', encoding='utf-8-sig') as f:
                input_conflict = json.load(f)
            self_logging(f"*** Save {session} input_conflict ***")
        self_logging(f"input_conflict: {len(input_conflict)}")

        if args.topk == -1:
            input_conflict = [item for item in input_conflict if item['predicted_label']['label'] == 'CONTRADICTION' and item['predicted_label']['score'] >= args.threshold]
        else: 
            filtered_data = [(index, item) for index, item in enumerate(input_conflict) if item["predicted_label"]["label"] == "CONTRADICTION"]
        
            grouped_data = defaultdict(list)
            for index, item in filtered_data:
                group_key = (item["episode_index"], item["speaker"])
                grouped_data[group_key].append((index, item))

            result = []
            for (episode, speaker), items in grouped_data.items():
                sorted_items = sorted(items, key=lambda x: x[1]["predicted_label"]["score"], reverse=True)
                top_items = sorted_items[:args.topk]
                
                top_items_in_original_order = sorted(top_items, key=lambda x: x[0])
                result.extend([item[1] for item in top_items_in_original_order])

            input_conflict = result

        self_logging(f"input_conflict: {len(input_conflict)}")

        os.makedirs('./output/pkb_filtered', exist_ok=True)
        if not os.path.exists('./output/pkb_filtered/{}_PKB_filtered_{}.json'.format(session, args.save_memo)):
            PKB_filtered_json = {}

            pkb_make_start = time.time()

            total = len(input_expanded_filtered.index)
            with Pool(processes=args.pool) as pool:
                tasks = [(episode_idx, input_expanded_filtered, input_conflict) for episode_idx in input_expanded_filtered.index]
                results = pool.starmap(process_episode, tasks)
                
                for episode_idx, episode_data in results:
                    PKB_filtered_json[episode_idx] = episode_data

            def key_sort(item):
                key = item[0]
                num = int(key.split('_')[-1])
                return num

            PKB_filtered_json = dict(sorted(PKB_filtered_json.items(), key=key_sort))

            pkb_make_finish = time.time()
            self_logging("\n")
            self_logging(f'time: {(pkb_make_finish - pkb_make_start)/ 60}분')
            
            with open('output/pkb_filtered/{}_PKB_filtered_{}.json'.format(session, args.save_memo), 'w') as json_file:
                json.dump(PKB_filtered_json, json_file, indent=4)
            self_logging(f"*** Save {session} PKB_filtered_json ***")
        else:
            with open('output/pkb_filtered/{}_PKB_filtered_{}.json'.format(session, args.save_memo), 'r', encoding='utf-8-sig') as f:
                PKB_filtered_json = json.load(f)
            self_logging(f"*** Load {session} PKB_filtered_json : output/pkb_filtered/{session}_PKB_filtered_{args.save_memo}.json ***")
            
        if session == sessions[0]: 
            pass
        else: 
            if not os.path.exists(input_conflict_intersession_filepath):
                self_logging(f'*** {sessions[sessions.index(session)-1]} 과 {session} 의 inter-session conflict 구하는 중 ***')
                if mp_spawn == False:
                    mp_spawn = True
                if models is None:
                    gpu_list = ast.literal_eval(args.gpu_list) 
                    multi_gpu_idx = get_least_used_gpu_multiple(gpu_list, args.num_gpu)
                    
                    if model is not None:
                        del model
                    models = [pipeline("text-classification", model="roberta-large-mnli", device=i, batch_size=batch_size) for i in multi_gpu_idx]
                intersession_conflict = comet_intersession_dnli_prediction(models, tokenizer, multi_gpu_idx, input_expanded_filtered_filepath, conflict_persona_graph.PKB, sessions[sessions.index(session)-1], args, persona_type="comet")
                self_logging(f'*** Save {session} intersession_conflict ***')
                with open(input_conflict_intersession_filepath, 'w') as json_file:
                    json.dump(intersession_conflict, json_file, indent=4)
            else:
                self_logging(f'*** Load {session} intersession_conflict : {input_conflict_intersession_filepath} ***')
                with open(input_conflict_intersession_filepath, 'r', encoding='utf-8') as f:
                    intersession_conflict = json.load(f)
                    
            if args.topk == -1: 
                intersession_conflict = [item for item in intersession_conflict if item['predicted_label']['label'] == 'CONTRADICTION' and item['predicted_label']['score'] >= args.threshold]
            else: 
                filtered_data = [(index, item) for index, item in enumerate(intersession_conflict) if item["predicted_label"]["label"] == "CONTRADICTION"]

                grouped_data = defaultdict(list)
                for index, item in filtered_data:
                    group_key = (item["episode_index"], item["speaker"])
                    grouped_data[group_key].append((index, item))

                result = []
                for (episode, speaker), items in grouped_data.items():
                    sorted_items = sorted(items, key=lambda x: x[1]["predicted_label"]["score"], reverse=True)
                    top_items = sorted_items[:args.topk]
                    
                    top_items_in_original_order = sorted(top_items, key=lambda x: x[0])
                    result.extend([item[1] for item in top_items_in_original_order])
                
                intersession_conflict = result

        if intersession_conflict is not None:  # session 2,3,4
            input_conflict_total = input_conflict + intersession_conflict
        elif intersession_conflict is None: # session 1
            input_conflict_total = input_conflict
        
        if args.topk == -1: 
            input_conflict_total = [item for item in input_conflict_total if item['predicted_label']['label'] == 'CONTRADICTION' and item['predicted_label']['score'] >= args.threshold]
        else:
            filtered_data = [(index, item) for index, item in enumerate(input_conflict_total) if item["predicted_label"]["label"] == "CONTRADICTION"]

            grouped_data = defaultdict(list)
            for index, item in filtered_data:
                group_key = (item["episode_index"], item["speaker"])
                grouped_data[group_key].append((index, item))

            result = []
            for (episode, speaker), items in grouped_data.items():
                sorted_items = sorted(items, key=lambda x: x[1]["predicted_label"]["score"], reverse=True)
                top_items = sorted_items[:args.topk]
                
                top_items_in_original_order = sorted(top_items, key=lambda x: x[0])
                result.extend([item[1] for item in top_items_in_original_order])

            input_conflict_total = result
        
        with open(input_conflict_total_filepath, 'w', encoding='utf-8') as f:
            json.dump(input_conflict_total, f) # conflict rg setting file
        
        self_logging(f'*** PKB update by removing node with conflict graph ***')

        conflict_persona_graph.graph = {}
        conflict_persona_graph.load_edges_from_file(input_conflict_total)
        passed = conflict_persona_graph.test_persona_in_dialog_tree(input_conflict_total)
        
        if os.path.exists('output/pkb_updated/{}_PKB_updated_{}.json'.format(session, args.save_memo)):
            self_logging(f'*** Load {session} conflict_persona_graph.PKB : output/pkb_updated/{session}_PKB_updated_{args.save_memo}.json ***')
            with open('output/pkb_updated/{}_PKB_updated_{}.json'.format(session, args.save_memo), 'r', encoding='utf-8') as json_file:
                conflict_persona_graph.PKB = json.load(json_file)
            
            self_logging(f'*** Load {session} conflict_persona_graph.persona_to_text : data/dialogue_tree_new/persona_to_text_{session}_{args.save_memo}.pkl ***')
            with open(f'data/dialogue_tree_new/persona_to_text_{session}_{args.save_memo}.pkl', 'rb') as file:
                conflict_persona_graph.persona_to_text = pickle.load(file)
        else:       
            if mp_spawn:
                exit()

            def worker(speaker, episode_index):
                return conflict_persona_graph.fast_chatgpt_multiprocessing(session, episode_index, speaker, args)

            step_count_zero = 0
            cost = 0
            results = []
            start_time = time.time()
            for speaker in [1, 2]:
                with Pool(processes=args.pool) as pool:
                    partial_worker = partial(worker, speaker)
                    
                    for result in pool.imap(partial_worker, input_expanded.index):
                        results.append(result)
                        cost += result['cost']
                        step_count_zero += result['step_count_zero']

            self_logging(f'{session} time cost: {(time.time() - start_time)/60} min')
            self_logging(f'{session} money cost: ${cost}')

            experiment_total_cost += cost
            
            self_logging(f"Money total used: ${experiment_total_cost}")


            del results

            conflict_persona_graph.graph = {}
            conflict_persona_graph.load_edges_from_file(input_conflict_total)
            for speaker in [1, 2]:
                for episode_index in tqdm(input_expanded.index):
                    result_dict = conflict_persona_graph.node_removal_simulations(session, episode_index, speaker, args)

            with open('output/pkb_updated/{}_PKB_updated_{}.json'.format(session, args.save_memo), 'w') as json_file:
                json.dump(conflict_persona_graph.PKB, json_file, indent=4)
            self_logging(f'*** Save {session} conflict_persona_graph.PKB ***')

            with open(f'data/dialogue_tree_new/persona_to_text_{session}_{args.save_memo}.pkl', 'wb') as file:
                pickle.dump(conflict_persona_graph.persona_to_text, file)

            self_logging(f'*** Save {session} conflict_persona_graph.persona_to_text ***')
            
        conflict_persona_graph.PKB = deep_update(conflict_persona_graph.PKB, PKB_filtered_json)
        
        if os.path.exists(output_PKB_filepath):
            pass
        else:
            self_logging(f'*** Save {session} UPDATED_PKB ***')
            with open(output_PKB_filepath, 'w') as json_file:
                json.dump(conflict_persona_graph.PKB, json_file, indent=4)
        
        if session == sessions[-1]: 
            end_time = time.time()
            self_logging(f"Used time: {(end_time - start_time)/60} min")
            self_logging(f"Used money: ${experiment_total_cost}")
            self_logging(f'*** Finished ***')

            exit()
