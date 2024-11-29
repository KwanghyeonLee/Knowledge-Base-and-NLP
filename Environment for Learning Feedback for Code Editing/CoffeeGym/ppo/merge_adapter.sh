sft_model_checkpoint=$1
ppo_adapter_checkpoint=$2
merged_checkpoint=$3
python src/research_projects/stack_llama/scripts/merge_peft_adapter.py --output_name=$output_name --base_model_name=$sft_model_checkpoint --adapter_model_name=$ppo_adapter_checkpoint
 