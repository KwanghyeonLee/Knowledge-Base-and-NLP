import json
import yaml
from functools import wraps
from pathlib import Path


def get_project_path():
    return Path(__file__).parents[1]


def ensure_absolute_path(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            path = Path(args[0])
            if not path.is_absolute():
                new_path = get_project_path() / path
                args = (new_path,) + args[1:]
        return func(*args, **kwargs)

    return wrapper


@ensure_absolute_path
def find_file_extension_files(directory, extension):
    directory_path = Path(directory)
    files = list(directory_path.rglob(f'*.{extension}'))
    return [str(file) for file in files]


@ensure_absolute_path
def get_output_path(file_path, extension) -> Path:
    input_path = Path(file_path)
    file_path = input_path.with_suffix(f'.{extension}')
    return file_path


@ensure_absolute_path
def read_text_file(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.exists() and file_path.is_file():
        with file_path.open(mode='r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        print(f"File '{file_path}' does not exist or is not a regular file.")
        return None

@ensure_absolute_path
def read_yaml_file(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.exists() and file_path.is_file():
        with file_path.open(mode='r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        return content
    else:
        print(f"File '{file_path}' does not exist or is not a regular file.")
        return None

@ensure_absolute_path
def read_json_file(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.exists() and file_path.is_file():
        with file_path.open(mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
    else:
        print(f"File '{file_path}' does not exist or is not a regular file.")
        return None

def load_prompt(model_name: str):
    prompt_path = get_project_path() / 'prompts' / f'{model_name}.yaml'
    return read_yaml_file(prompt_path)



@ensure_absolute_path
def save_json_file(file_path, data):   
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Ensure the directory exists before writing the file
    file_path.parent.mkdir(parents=True, exist_ok=True)
 
    with file_path.open(mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@ensure_absolute_path
def save_jsonl_file(file_path, data):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Ensure the directory exists before writing the file
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode='w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


@ensure_absolute_path
def collect_items_from_jsonl(dir_path):
    file_paths = find_file_extension_files(dir_path, 'json')
    collected_items = []

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                collected_items.append(item)

    return collected_items


@ensure_absolute_path
def find_file(root_path, file_name):
    root = Path(root_path)
    for file in root.glob('**/*'):
        if file.name == file_name:
            return file.resolve()
        
