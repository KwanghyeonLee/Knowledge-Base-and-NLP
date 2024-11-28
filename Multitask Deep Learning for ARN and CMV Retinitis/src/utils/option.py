import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', help='train or test')
    parser.add_argument('--model_name', type=str, default='ADMTL', help='model name[STL, FSMTL, SPMTL, ADMTL]')
    parser.add_argument('--class_type', type=int, default=1, help='class type for STL model [1, 2]')
    
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    
    
    ## Hyper parameters
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    
    parser.add_argument('--alpha', type=float, default=0.125, help='alpha of task 1 loss')
    parser.add_argument('--beta', type=float, default=1.0, help='beta of total Task loss')
    parser.add_argument('--lambda_adv', type=float, default=0.05, help='lambda for adversarial loss')
    parser.add_argument('--gamma', type=float, default=0.01, help='gamma for diff loss')
    
    # parser.add_argument('--save_model_path', type=str, default='./model/ADMTL/', help='save model path')
    parser.add_argument('--model_dir', type=str, default='./model', help="Model's directory")
    
    parser.add_argument('--logger_dir', type=str, default='./log/logger', help="Logger's directory")
    parser.add_argument('--log_tb_dir', type=str, default='./log/tensorboard', help='tensorboard log dir')
    
    parser.add_argument('--do_test', action='store_true', help='do test')
    # For SHAP
    parser.add_argument('--device_cpu', type=bool, default=False, help='device cpu')
    args = vars(parser.parse_args())
    return args

