from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-1.3b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    freeze_LM: Optional[bool] = field(
        default=True,
        metadata={"help": "Freeze the LM weights."}
    )
    model_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Use 4bit model."}
    )
    img_token_num: Optional[int] = field(
        default=2,
        metadata={"help": "Number of image tokens."}
    )
    img_input_channel: Optional[int] = field(
        default=15,
        metadata={"help": "Number of image input channels."}
    )
    resnet_depth: Optional[int] = field(
        default=50,
        metadata={"help": "Resnet depth."}
    )
    resnet_fc_dim: Optional[int] = field(
        default=2048,
        metadata={"help": "Resnet fc dim."}
    )
    
@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=600,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=512,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    img_type: str = field(
        default='parcel',
        metadata={"help": "Type of image to use. Should be one of `mr` or `parcel`"}
    )



@dataclass
class TrainingArguments:
    wandb_project: Optional[str] = field(
        default="MedicalCoT"
    )
    wandb_name: Optional[str] = field(
        default="opt-1.3b"
    )
    save_dir: Optional[str] = field(
        default="your/pytorch-lightning/checkpoint/dir"
    )
    devices: Optional[List[int]] = field(
        default_factory=lambda: [0]
    )
    max_epochs: Optional[int] = field(
        default=100
    )
    do_train: Optional[bool] = field(
        default=True
    )
    do_test: Optional[bool] = field(
        default=True
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4
    )
    predict_with_generate: Optional[bool] = field(
        default=False
    )
    lr: Optional[float] = field(
        default=0.0002
    )
    max_lr: Optional[float] = field(
        default=0.0002
    )
    step_size_up: Optional[int] = field(
        default=1000
    )
    scheduler_type: Optional[str] = field(
        default='ReduceLROnPlateau'
    )
    scheduler_mode: Optional[str] = field(
        default='min'
    )
    factor: Optional[float] = field(
        default=0.1
    )
    patience: Optional[int] = field(
        default=2
    )
    save_prediction_file: Optional[str] = field(
        default="/dir/to/save/prediction/file"
    )
    strategy: Optional[str] = field(
        default='deepspeed_stage_2'
    )


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=700,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=0.7)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0) 