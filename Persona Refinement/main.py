import subprocess
import os

def run_script(script_name_with_args):
    args_list = script_name_with_args.split(' ')
    if args_list[0] == 'cd':
        result = subprocess.run(args_list, shell=True)
    else:
        result = subprocess.run(args_list)

def run_script_error(script_name_with_args):
    args_list = script_name_with_args.split(' ')
    if args_list[0] == 'cd':
        result = subprocess.run(args_list, shell=True)
    else:
        result = subprocess.run(args_list) 
    
    if result.returncode != 0:
        raise Exception(f"Error executing script: {script_name_with_args}")

def main():
    
    # Caffein memory refinement setting
    your_path = ''
    
    PROJECT_DIR = f'{your_path}/Caffeine'

    parsing_test = ''
    
    description = 'experiment_001'
    save_memo = 'experiment_001'

    model_type = 'chat-gpt'
    max_token = 500
    temperature = 1.0
    stop_seq = '\n\n\n'
    assess_result_dir = f'{PROJECT_DIR}/output/assessment_result'
    prompt_file_final = f'{PROJECT_DIR}/prompts/refine_5shot.txt'
    expansion_type = 'comet'
    num_gpu = 8
    batch_size = 512
    threshold = 0.8
    log_dir = f'{PROJECT_DIR}/log'
    epi_num= '_episode_10'
    pool = 8
    key_file = ''
    gpu_list = '[0,1,2,3,4,5,6,7]'

    expansion = 'comet'
    input_expanded_filepath_before = f'data/comet_expansion/comet_'
    input_expanded_filepath_after = f"_beam_{epi_num}"

    last_session = 'session4'
    output_PKB_filepath = f'output/final_pkb_updated{epi_num}/final_pkb_{last_session}{epi_num}_{save_memo}.json'

    scripts = []
    scripts.append(f"/home/common/miniconda3/envs/Caffeine/bin/python src/dialogue_tree.py --epi_num {epi_num} --input_expanded_filepath_before {input_expanded_filepath_before} --input_expanded_filepath_after {input_expanded_filepath_after} --expansion {expansion}")

    # Caffein memory refinement 
    scripts.append(
        f"/home/common/miniconda3/envs/Caffeine/bin/python src/pkb_update_graph.py --description {description} --save_memo {save_memo} --model_type {model_type} --max_token {max_token} --temperature {temperature} --stop_seq {stop_seq} --assess_result_dir {assess_result_dir} --prompt_file_final {prompt_file_final} --topk {topk} --expansion_type {expansion_type} --batch_size {batch_size} --threshold {threshold} --log_dir {log_dir} --epi_num {epi_num} --pool {pool} --mp_spawn --num_gpu {num_gpu} --parsing_test {parsing_test} --use_source --key_file {key_file} --gpu_list {gpu_list} --input_expanded_filepath_before {input_expanded_filepath_before} --input_expanded_filepath_after {input_expanded_filepath_after} --expansion {expansion}",
    )

    for script in scripts:
        if os.path.exists(output_PKB_filepath):
            print(f'{output_PKB_filepath}')
            print("Memory refinement is done.")
            break
        
        run_script(script)

        print("="*30)
        print("\n")
        print("\n")
        print("Restart for multiprocessing.")
        print("\n")
        print("\n")
        print("="*30)

    # Response generation setting
    PROJECT_DIR = f'{your_path}/Caffeine/Response_generation'
    MAIN = f'{your_path}/Caffeine'

    api_key, org_id = 'your_api_key', 'your_organization'

    call_amount = 300
    
    NoRetrieve = False
    Retrieve = True
    sessions_num = [1,2,3,4]
    
    RG_step = []

    os.makedirs(f"{PROJECT_DIR}/data/msc/preprocessed/updated_pkb/{save_memo}", exist_ok=True)
    
    for top_K in [20]:
        for i in sessions_num:
            final_pkb_file_name = f'final_pkb_session{i}{epi_num}_{save_memo}.json'
            
            # preprocess
            RG_step.append(
                f"/home/common/miniconda3/envs/Caffeine_lang/bin/python {PROJECT_DIR}/utils/make_pkb_2_rg_agg.py --pkb_path {MAIN}/output/final_pkb_updated{epi_num}/{final_pkb_file_name} --rg_format_file_path {PROJECT_DIR}/data/msc/preprocessed/response_generation/test.json --save_dir {PROJECT_DIR}/data/msc/preprocessed/updated_pkb/{save_memo}"
            )
            
            if NoRetrieve:
                RG_step.append(
                    f"/home/common/miniconda3/envs/Caffeine_lang/bin/python {PROJECT_DIR}/src/inference_chatgpt.py --api_key {api_key} --org_id {org_id} --input_path {PROJECT_DIR}/data/msc/preprocessed/updated_pkb/{save_memo}/{final_pkb_file_name.split('final_')[1]} --prompt {PROJECT_DIR}/prompts/rg.yaml --prompt_key base_rg --save_dir {PROJECT_DIR}/result/response_generation/{save_memo}/session{i}_{save_memo}/preprocessed --ordered_session {i} --call_amount {call_amount}")

            if Retrieve:
                RG_step.append(
                    f"/home/common/miniconda3/envs/Caffeine/bin/python {PROJECT_DIR}/utils/retrieve_parallel.py --input_path {PROJECT_DIR}/data/msc/preprocessed/updated_pkb/{save_memo}/pkb_session{i}{epi_num}_{save_memo}.json --output_dir {PROJECT_DIR}/data/msc/retrieve/contriever/updated_pkb_session{i}{epi_num}_{save_memo} --retriever contriever --top_k {top_K} --gpu_list {gpu_list}"
                )                

                RG_step.append(
                    f"/home/common/miniconda3/envs/Caffeine_lang/bin/python {PROJECT_DIR}/src/inference_chatgpt.py --api_key {api_key} --org_id {org_id} --input_path {PROJECT_DIR}/data/msc/retrieve/contriever/updated_pkb_session{i}{epi_num}_{save_memo}/{top_K}/pkb_session{i}{epi_num}_{save_memo}.json --prompt {PROJECT_DIR}/prompts/rg.yaml --prompt_key base_rg --save_dir {PROJECT_DIR}/result/response_generation/{save_memo}/session{i}_{save_memo}/top_{top_K}_retrieved --ordered_session {i} --use_retrieval --call_amount {call_amount}"
                )

        ## eval
        if NoRetrieve:
            for i in sessions_num:
                # evaluation without nltk
                RG_step.append(
                    f"/home/common/miniconda3/envs/Caffeine_lang/bin/python {PROJECT_DIR}/src/eval.py --input_path {PROJECT_DIR}/result/response_generation/{save_memo}/session{i}_{save_memo}/preprocessed.json --task direct --session_id {i} --sheet_name wo_nltk --experiment No_Retrieval --method {save_memo}"
                )

        if Retrieve:
            for i in sessions_num:
                # evaluation without nltk
                RG_step.append(
                    f"/home/common/miniconda3/envs/Caffeine_lang/bin/python {PROJECT_DIR}/src/eval.py --input_path {PROJECT_DIR}/result/response_generation/{save_memo}/session{i}_{save_memo}/top_{top_K}_retrieved.json --task direct --session_id {i} --sheet_name wo_nltk --experiment top_{top_K} --method {save_memo}"
                )

    
    for script in RG_step:

        run_script_error(script)

        print("="*30)
        print("\n")
        print(f"Done: {script}")
        print("\n")
        print("="*30)

if __name__ == '__main__':
    main()
