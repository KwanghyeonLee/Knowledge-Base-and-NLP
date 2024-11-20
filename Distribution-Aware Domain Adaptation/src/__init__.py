import torch
from transformers import logging
import os
"""
“highest”, float32 matrix multiplications use the float32 datatype for internal computations.

“high”, float32 matrix multiplications use the TensorFloat32 or bfloat16_3x datatypes for internal computations, 
if fast matrix multiplication algorithms using those datatypes internally are available. 
Otherwise float32 matrix multiplications are computed as if the precision is “highest”.

“medium”, float32 matrix multiplications use the bfloat16 datatype for internal computations, 
if a fast matrix multiplication algorithm using that datatype internally is available. 
Otherwise float32 matrix multiplications are computed as if the precision is “high”.
"""

os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
logging.set_verbosity_error()
torch.set_float32_matmul_precision("high")
