import subprocess

def get_least_used_gpu(gpu_indices):

    gpu_memory_map = {gpu: 0 for gpu in gpu_indices}
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    for line in result.strip().split('\n'):
        index, mem = map(int, line.split(', '))
        if index in gpu_indices:
            gpu_memory_map[index] = mem

    least_used_gpu = min(gpu_memory_map, key=gpu_memory_map.get)
    return least_used_gpu


def get_least_used_gpu_multiple(gpu_indices, N):
    
    gpu_memory_map = {gpu: 0 for gpu in gpu_indices}
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    for line in result.strip().split('\n'):
        index, mem = map(int, line.split(', '))
        if index in gpu_indices:
            gpu_memory_map[index] = mem

    sorted_gpus = sorted(gpu_memory_map.items(), key=lambda x: x[1])
    least_used_gpus = [gpu[0] for gpu in sorted_gpus[:N]]
    
    return least_used_gpus
