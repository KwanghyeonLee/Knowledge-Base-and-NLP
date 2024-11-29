import subprocess
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import torch

TEST_CASE_PATH = "PATH_TO_TEST_CASES"
def run_code(code, input_data):
    process = subprocess.Popen(
        ['python', '-c', code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Join inputs with newline and encode
    # input_data = input_data.replace("\r\n", "\n")
    input_data = input_data.encode()
    
    try:
        stdout_data, stderr_data = process.communicate(input=input_data, timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        print("The subprocess exceeded the time limit and was killed.")
        return None

    if process.returncode != 0:
        # There was an error
        print(code)
        print(input_data)
        print(Exception(f"Error executing code:\n{stderr_data.decode('utf-8')}"))
        return None
    
    return stdout_data.decode('utf-8')

def score_output(predictions, references):
    # Handle None values and perform string operations
    predictions = [p.replace("\r\n", "\n").strip("\n") if p is not None else None for p in predictions]
    references = [r.replace("\r\n", "\n").strip("\n") if r is not None else None for r in references]
    
    # Calculate score
    score = sum([p == r for p, r in zip(predictions, references)]) / len(predictions) if predictions else 0
    
    # Print for debugging
    print(predictions)
    print(references)
    print(score)

    return torch.tensor([score])

def calculate_test_score(refined_codes, problem_ids):
    test_case_list = []
    for pi in problem_ids:
        with open(os.path.join(TEST_CASE_PATH, f"{pi}.json"),"r") as f:
            cur_test_case = json.load(f)['data'][0]['test_cases']
            test_case_list.append(cur_test_case)
    
    test_case_predictions = []
    scores = []
    for ci, c in enumerate(tqdm(refined_codes)): 
        print(problem_ids[ci])
        cur_testcases = test_case_list[ci]
        cur_inputs = [t['input'] for t in cur_testcases]
        cur_outputs = [run_code(c, ti) for ti in cur_inputs]
        cur_references =[t['output'] for t in cur_testcases]
        cur_score = score_output(cur_outputs, cur_references)
        scores.append(cur_score)
    
        print("="*10) 
        print(sum(scores)/len(scores))
        print("="*10) 
    return scores 
    



if __name__ == "__main__":
    trainset = load_dataset("DATASET")['eval']
    refined_codes = [d['correct_code'] for d in trainset]
    problem_ids = [d['problem_id'] for d in trainset]
    scores = calculate_test_score(refined_codes, problem_ids) 
    print(sum(scores)/len(problem_ids))