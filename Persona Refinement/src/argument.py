import argparse

MODEL_MAPPING = {
    'text-davinci-003': 'text-davinci-003',
    'text-curie-001': 'text-curie-001',
    'chat-gpt': 'gpt-3.5-turbo',
    'gpt-4' : 'gpt-4'
}

def load_parser_and_args():
    parser = argparse.ArgumentParser()
    # GPT args
    parser.add_argument('--model_type', type=str, default='chat-gpt', required=True)
    parser.add_argument('--max_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--stop_seq', type=str, default='\n\n')

    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--dialog_tree', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--prompt_dir', type=str, default='')
    parser.add_argument('--prompt_file_final', type=str, default='')
    parser.add_argument('--assess_result_dir', type=str, default='')
    
    parser.add_argument('--log_level', type=int, default=-1)
    parser.add_argument('--save_memo', type=str, default='', help="To distinguish file names")
    parser.add_argument('--topk', type=int, default=-1,help="-1 means threshold method")
    parser.add_argument('--threshold', type=float, default=0.8, help="conflict filtering threshold")
    
    parser.add_argument("--expansion_type", type=str, default='comet', help="knowledge expansion type")
    parser.add_argument("--batch_size", type=int, default=4069, help="nli model batch size")
    parser.add_argument("--epi_num", type=str, default='', help="The number of episode to experiment")
    parser.add_argument("--description", type=str, default='', help="Description of experiment")
    parser.add_argument('--pool', type=int, default=4, help="The number of multiprocessing pool")
    parser.add_argument('--num_gpu', type=int, default=2)

    parser.add_argument("--mp_spawn", action="store_true", help="For Multi processing")
    parser.add_argument("--use_source", action="store_true", help="Use source persona in prompt")
    parser.add_argument('--key_file', type=str, required=True)
    parser.add_argument('--gpu_list', type=str, default='gpu list')
    parser.add_argument('--category', type=str, default="['[Resolution]','[Disambiguation]']")
    parser.add_argument('--node_removal', type=str, default="Sum")

    parser.add_argument('--input_expanded_filepath_before', type=str, required=True)
    parser.add_argument('--input_expanded_filepath_after', type=str, required=True)
    parser.add_argument('--expansion', type=str, required=True)

    args = parser.parse_args()
    args.model_name_or_path = MODEL_MAPPING[args.model_type]

    return parser, args