from helper import get_embeddings, get_param_numbers
from os.path import join as path_join
from models import TestNetAttention, RNNJointAttention, JordanRNNJointAttention, TranslatorJointAttention
from baselines.models import StructuralRNN


import argparse
import numpy as np


def __pars_args__():
    parser = argparse.ArgumentParser(description='Guided attention model')
    parser.add_argument("--data_dir", "-d_dir", type=str, default="sintetic", help="Directory containing dataset file")
    parser.add_argument("--dataset_prefix", type=str, default="tr_", help="Prefix for the dataset")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_dataset.bin", help="Train file name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str, default="eval_dataset.bin", help="Eval file name")
    parser.add_argument("--test_file_name", "-test_fn", type=str, default="test_dataset.bin", help="Test file name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=30, help='Batch size for eval.')

    parser.add_argument('--input_dim', type=int, default=1, help='Embedding size.')
    parser.add_argument('--hidden_dim', type=int, default=5, help='Hidden state memory size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of rnn layers.')
    parser.add_argument('--time_windows', type=int, default=10, help='Attention time windows.')
    parser.add_argument('--max_neighbors', "-m_neig", type=int, default=4, help='Max number of neighbors.')
    parser.add_argument('--output_dim', type=int, default=1, help='output size.')
    parser.add_argument('--drop_prob', type=float, default=0.1, help="Keep probability for dropout.")
    parser.add_argument('--temp', type=float, default=0.45, help="Softmax temperature")
    parser.add_argument('--n_head', type=int, default=6, help="attention head number")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=102, help="Iteration number.")


    parser.add_argument('--eval_step', type=int, default=10, help='How often do an eval step')
    parser.add_argument('--save_rate', type=float, default=0.9, help='How often do save an eval example')
    return parser.parse_args()



if __name__ == "__main__":
    args = __pars_args__()
    input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor = get_embeddings(path_join("..", "data", args.data_dir), prefix=args.dataset_prefix)
    model = RNNJointAttention(args.input_dim, args.hidden_dim, args.output_dim, args.n_head, args.time_windows, dropout_prob=args.drop_prob, temperature=args.temp)

    # model = StructuralRNN(args.input_dim, args.hidden_size, args.output_size, args.num_layers, args.max_neighbors, input_embeddings.size(1),
    #                       dropout_prob=args.drop_prob)

    n_params = get_param_numbers(model)
    print(n_params)