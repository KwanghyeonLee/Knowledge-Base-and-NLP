import os
import json
import logging
import datetime
import time
import openai
import signal
import re
import ast

# set logger
logger = logging.getLogger('logger')


def openai_login(args):
    with open(f'./utils/{args.key_file}.json') as f:
        keys = json.load(f)
    openai.organization = keys[0] 
    openai.api_key = keys[1] 
    

def init_logger(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    # save log files
    handler = logging.FileHandler(os.path.join(args.log_dir, '{:%Y-%m-%d-%H:%M:%S}.log'.format(datetime.datetime.now())), encoding='utf=8')
    logger.addHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.log_level in [-1, 0] else logging.WARN,
    )
    logger.warning(args)


class GPT3(object):
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.frequency_penalty = args.frequency_penalty
        self.presence_penalty = args.presence_penalty
        self.stop_seq = args.stop_seq
        self.org_key = 0
        self.private_key = 1

    def chatgpt_inference(self, prompt, args):
        time.sleep(0.1)
        total_cost = 0
        cost_times = 0

        def handler(signum, frame):
            raise Exception("ChatGPT time is up!")
        
        signal.signal(signal.SIGALRM, handler)

        wait_time = 30
        while True:
            try:
                signal.alarm(60) 
                
                output = openai.ChatCompletion.create(                    
                    model=self.model_name,
                    messages=[
                            {"role": "user", "content": prompt},
                        ],
                    n=1, 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop_seq
                )
                signal.alarm(0) 

                output_string = output['choices'][0]['message']['content'] 
                output_string = output_string.strip() 
            
                prompt_token_used = output['usage']['prompt_tokens']
                completion_token_used = output['usage']['completion_tokens']
                total_cost += (completion_token_used / 1000 * 0.002) + (prompt_token_used / 1000 * 0.0015) # gpt-3.5-turbo
                #total_cost += token_used / 1000 * 0.06  # gpt-4

                total_cost += (prompt_token_used / 1000 * 0.0015) * cost_times 

                return output_string, total_cost
            
            # connection error, timeout error.
            except Exception as e:
                print(e)
                if str(e) == "ChatGPT time is up!":
                    cost_times += 1
                time.sleep(wait_time)
                wait_time += 5
                print(f'ChatGPT query restart!: {wait_time}')
                continue


def parse_personas(text):
    persona_sentences = re.split('Persona \d+: ', text)[1:]  
    
    result = []
    
    for sentence in persona_sentences:
        result.append(sentence.strip())
    
    return result


def extract_personas(text):
    # Extracting sentences after "Persona 1:" and "Persona 2:"
    persona_1_match = re.search(r"Persona 1: ([^\.]+)\.", text)
    persona_2_match = re.search(r"Persona 2: ([^\.]+)\.", text)

    persona_1 = persona_1_match.group(1) if persona_1_match else "Not Found"
    persona_2 = persona_2_match.group(1) if persona_2_match else "Not Found"
    
    return persona_1, persona_2


def classifier(chatgpt_output, node_1, node_2, category=['[Resolution]', '[Disambiguation]']):
    category = ast.literal_eval(category)
    if '[Resolution]' in category and '[Disambiguation]' in category:
        try:
            if '[NO_CONFLICT]' in chatgpt_output:
                refined_personas = [node_1, node_2]
            elif '[Resolution]' in chatgpt_output:
                output = chatgpt_output.split('[Resolution]')[1].strip().lstrip(": ") 
                
                refined_personas  = [output]
            elif '[Disambiguation]' in chatgpt_output:
                personas = extract_personas(chatgpt_output)
                if "Not Found" in personas:
                    refined_personas = [node_1, node_2]
                else:
                    refined_personas = personas
            else:
                
                refined_personas = [node_1, node_2]

        except:
            refined_personas = [node_1, node_2]

    elif '[Resolution]' in category and '[Disambiguation]' not in category:
        try:
            if '[Resolution]' in chatgpt_output:
                output = chatgpt_output.split('[Resolution]')[1].strip().lstrip(": ") #[2:]
                
                refined_personas  = [output]
            else:
                
                refined_personas = [node_1, node_2]
        except: 
            refined_personas = [node_1, node_2]
        
    elif '[Resolution]' not in category and '[Disambiguation]' in category:
        try:
            if '[Disambiguation]' in chatgpt_output:
                personas = extract_personas(chatgpt_output)
                if "Not Found" in personas:
                    refined_personas = [node_1, node_2]
                else:
                    refined_personas = personas
            else:
                
                refined_personas = [node_1, node_2]

        except: 
            refined_personas = [node_1, node_2]

    return refined_personas


def cot_reasoning(node_1, node_2, evidence_1, evidence_2, args, episode_index, speaker, asses_idx, session, src_node1, src_node2):  
     
    if not os.path.exists(args.assess_result_dir):
        os.makedirs(args.assess_result_dir)


    with open(args.prompt_file_final, 'r') as f:
        prompt_final = f.read()
    
    # load model
    model = GPT3(args)
    # openai login 
    openai_login(args)

    # if 'A' in speaker:
    if str(speaker) == '1':
        speaker_map = 'Speaker1'
    # elif 'B' in speaker:
    elif str(speaker) == '2':
        speaker_map = 'Speaker2'
    
    try:
        
        model_input_3 = prompt_final % (node_1, ', '.join(evidence_1), node_2, ', '.join(evidence_2))
        
    except:
        model_input_3 = prompt_final % (node_1, evidence_1[0], node_2, evidence_2[0])
        
    # inference    
    chatgpt_output, cost = model.chatgpt_inference(model_input_3, args) 
    
    # classify
    refined_personas = classifier(chatgpt_output, node_1, node_2, args.category)


    os.makedirs(os.path.join(args.assess_result_dir, args.save_memo), exist_ok=True)
    os.makedirs(os.path.join(args.assess_result_dir, args.save_memo, session), exist_ok=True)
    if os.path.exists(os.path.join(args.assess_result_dir, args.save_memo, session, 'results_{}_{}_{}_{}.json'.format(args.model_type, session, speaker, episode_index))):
        with open(os.path.join(args.assess_result_dir, args.save_memo, session, 'results_{}_{}_{}_{}.json'.format(args.model_type, session, speaker, episode_index)), 'r', encoding='utf-8') as f:
            results = json.load(f)
            results[f'assesment_idx_{asses_idx}'] = {
                'assesment_idx': asses_idx,
                'episode_index': episode_index,
                'speaker':speaker,
                'node_1': node_1,
                'source_1':src_node1,
                'evidence_1': evidence_1,
                'node_2': node_2,
                'source_2':src_node2,
                'evidence_2': evidence_2,
                'chatgpt_output': chatgpt_output,
                'refined_personas' : refined_personas,
                'cost': cost,
                'prompt': model_input_3
            }
    else:
        results = {f'assesment_idx_{asses_idx}': {
            'assesment_idx': asses_idx,
            'episode_index': episode_index,
            'speaker':speaker,
            'node_1': node_1,
            'source_1':src_node1,
            'evidence_1': evidence_1,
            'node_2': node_2,
            'source_2':src_node2,
            'evidence_2': evidence_2,
            'chatgpt_output': chatgpt_output,
            'refined_personas' : refined_personas,
            'cost': cost,
            'prompt': model_input_3
        }}
    
    # save the results
    with open(os.path.join(args.assess_result_dir, args.save_memo, session, 'results_{}_{}_{}_{}.json'.format(args.model_type, session, speaker, episode_index)), 'w', encoding='utf-8') as f:
       json.dump(results, f, indent=4)
 
    return refined_personas, cost
    