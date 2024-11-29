import re

def extract_python_code(text):
    """
    Extracts Python code blocks from a given string that contain text enclosed within
    triple backticks labeled as 'python'.
    
    Parameters:
    - text (str): The input text containing Python code blocks.
    
    Returns:
    - list of str: A list of extracted code blocks.
    """
    try:
        patterns = [
        r"```python(.*?)'''", # Triple single quotes (general)
        r"```python(.*?)```", # Triple single quotes (general)
        r'```python(.*?)```',  # Triple backticks with 'python'
        r'```Python(.*?)```',
        r'```Python(.*?)\'\'\'',
        r'```(.*?)```',        # Triple backticks without a label
        r"'''python(.*?)'''",
        r"'''Python(.*?)'''",# Triple single quotes with 'python'
        r"'''(.*?)'''",        # Triple single quotes without a label
        r'```(.*?)```',        # Triple backticks (general)
        r"'''(.*?)'''",
        ] # Non-greedy match
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)  # Capture multiline code blocks
            if matches:
                return matches[0].strip()
        
        # Strip leading/trailing whitespace from each match and return
        return matches[0].strip()
    except:
        return text.split("```python\n")[-1].split("```")[0]
