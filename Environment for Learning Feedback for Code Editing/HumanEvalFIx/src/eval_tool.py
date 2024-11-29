import concurrent.futures
import time
import multiprocessing, time
import os
import sys

import json
import argparse
from tqdm import tqdm

# run execution in parallel



def execute_with_timeout(code, timeout):
    def target_func(code, queue):
        pre = queue.get() 
        res_dict = {}
        #time.sleep(2)
        try:
            exec(code, res_dict)
            #print(res_dict['predict'])
        except Exception as e:
            #print(e)
            # res_dict['predict'] = f"Error occurred: {e}"
            res_dict['predict'] = f"Error occurred"
        pre['predict'] = str(res_dict['predict'])
        queue.put(pre)

    pre = {'predict': None}
    queue = multiprocessing.Queue()
    queue.put(pre)
    process = multiprocessing.Process(target=target_func, args=(code, queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Execution timed out")
        
    
    return_val = queue.get()['predict']
    return return_val    

def evaluate_solutions(data_li, target_code, target_test):
    predict_li = []
    for di, data in enumerate(tqdm(data_li)): # iterate over progblem list        
        subdict = {} 
        subdict['task_id'] = data['task_id']
        prefix = data['prompt']
        str_code = data[target_code] #reference code
        example_test = data[target_test]
        
        try:
            a = execute_with_timeout(prefix +'\n'+str_code + '\n'+  example_test+  '\npredict = "correct"', timeout=5)
            # subdict['error_msg'] = a
            subdict['error_msg'] = "Error occurred"
        except TimeoutError as e:
            #print(f"Timeout occurred")
            subdict['error_msg'] = "Timeout occurred."

        predict_li.append(subdict)
    
    return predict_li

        


if __name__ == "__main__":
    with open('data/humaneval_fix_test_formatted_final.json', "r") as f:
        data_li = json.load(f)

    # run evaluation
    evaluate_solutions(data_li)