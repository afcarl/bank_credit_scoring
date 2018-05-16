from helper import CustomDataset, get_embeddings, get_customer_embeddings, ensure_dir, hookFunc
from worker import setup_model
from os.path import join as path_join
from torch.utils.data import DataLoader
from models import TestNetAttention, RNNJointAttention, JordanRNNJointAttention, TranslatorJointAttention
import torch.optim as optim
import torch

import argparse
import visdom
from datetime import datetime
import pickle



vis = visdom.Visdom(port=8080)
EXP_NAME = "exp-{}".format(datetime.now())



def __pars_args__():
    parser = argparse.ArgumentParser(description='Guided attention model')
    parser.add_argument("--data_dir", "-d_dir", type=str, default=path_join("data", "sintetic"), help="Directory containing dataset file")
    parser.add_argument("--dataset_prefix", type=str, default="simple_random_dynamic_", help="Prefix for the dataset")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_dataset", help="Train file name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str, default="eval_dataset", help="Eval file name")
    parser.add_argument("--test_file_name", "-test_fn", type=str, default="test_dataset", help="Test file name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=30, help='Batch size for eval.')

    parser.add_argument('--input_dim', type=int, default=35, help='Embedding size.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden state memory size.')
    parser.add_argument('--output_dim', type=int, default=1, help='output size.')
    parser.add_argument('--time_windows', type=int, default=10, help='Attention time windows.')
    parser.add_argument('--max_neighbors', "-m_neig", type=int, default=4, help='Max number of neighbors.')
    parser.add_argument('--drop_prob', type=float, default=0.1, help="Keep probability for dropout.")
    parser.add_argument('--temp', type=float, default=0.45, help="Softmax temperature")
    parser.add_argument('--n_head', type=int, default=4, help="attention head number")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=102, help="Iteration number.")


    parser.add_argument('--eval_step', type=int, default=10, help='How often do an eval step')
    parser.add_argument('--save_rate', type=float, default=0.9, help='How often do save an eval example')
    return parser.parse_args()



if __name__ == "__main__":
    args = __pars_args__()
    input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor = get_embeddings(args.data_dir, prefix=args.dataset_prefix)
    model = TranslatorJointAttention(args.input_dim, args.hidden_dim, args.output_dim, args.n_head, args.time_windows, dropout_prob=args.drop_prob, temperature=args.temp)

    train_dataset = CustomDataset(args.data_dir, args.dataset_prefix + args.train_file_name)
    eval_dataset = CustomDataset(args.data_dir, args.dataset_prefix + args.eval_file_name)
    test_dataset = CustomDataset(args.data_dir, args.dataset_prefix + args.test_file_name)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, drop_last=True)

    if args.use_cuda:
        model.cuda()

    model.reset_parameters()
    train_fn = setup_model(model, args.batch_size, args, True)
    eval_fn = setup_model(model, args.eval_batch_size, args, False)


    total_loss = []
    eval_number = 0
    eval_loss = []
    best_model = float("infinity")

    for i_iter in range(args.n_iter):
        iter_loss, _ = train_fn(train_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor)
        total_loss.append(iter_loss)

        print(iter_loss)

        # plot loss
        vis.line(
            Y=torch.FloatTensor(total_loss),
            X=torch.LongTensor(range(i_iter + 1)),
            opts=dict(
                    # legend=["loss", "penal", "only_loss"],
                    legend=["loss"],
                    title=model.name + " training loos",
                    showlegend=True),
            win="win:train-{}".format(EXP_NAME))

        if i_iter % args.eval_step == 0:
            iter_eval, saved_weights = eval_fn(eval_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor)
            eval_loss.append(iter_eval)

            vis.line(
                Y=torch.FloatTensor(eval_loss),
                X=torch.LongTensor(range(0, i_iter + 1, args.eval_step)),
                opts=dict(legend=["RMSE"],
                          title=model.name + " eval loos",
                          showlegend=True),
                win="win:eval-{}".format(EXP_NAME))

            pickle.dump(saved_weights, open(ensure_dir(path_join(args.data_dir, model.name, "{}saved_eval_iter-{}_temp-{}.bin".format(args.dataset_prefix, int(i_iter/args.eval_step), args.temp))), "wb"))

            if best_model > iter_eval:
                print("save best model")
                best_model = iter_eval
                torch.save(model, path_join(args.data_dir, "{}.pt".format(model.name)))

    # test performance
    model = torch.load(path_join(args.data_dir, "{}.pt".format(model.name)))
    test_fn = setup_model(model, args.eval_batch_size, args, False)

    iter_test, saved_weights = test_fn(test_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor)
    print("test RMSE: {}".format(iter_test))
    pickle.dump(saved_weights, open(ensure_dir(
        path_join(args.data_dir, model.name, "{}saved_test_temp-{}.bin".format(args.dataset_prefix, args.temp))), "wb"))