from helper import CDataset, get_embeddings, get_customer_embeddings, ensure_dir, hookFunc, DatasetInfo
from worker import setup_model
from os import path
from torch.utils.data import DataLoader
from models import TestNetAttention, RNNJointAttention, JordanRNNJointAttention, TranslatorJointAttention, TestSingleAttention
import torch
import numpy as np

import argparse
import visdom
from datetime import datetime
import pickle



vis = visdom.Visdom(port=8097,
                    use_incoming_socket=False)


def __pars_args__():
    parser = argparse.ArgumentParser(description='Guided attention model')
    parser.add_argument("--data_dir", "-d_dir", type=str, default="sintetic", help="Directory containing dataset file")
    parser.add_argument("--dataset_prefix", type=str, default="simple_random_neigh-100_rel-4_", help="Prefix for the dataset")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_dataset", help="Train file name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str, default="eval_dataset", help="Eval file name")
    parser.add_argument("--test_file_name", "-test_fn", type=str, default="test_dataset", help="Test file name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=50, help='Batch size for eval.')

    parser.add_argument('--input_dim', type=int, default=1, help='Embedding size.')
    parser.add_argument('--hidden_dim', type=int, default=5, help='Hidden state memory size.')
    parser.add_argument('--output_dim', type=int, default=1, help='output size.')
    parser.add_argument('--time_windows', type=int, default=10, help='Attention time windows.')
    parser.add_argument('--max_neighbors', "-m_neig", type=int, default=4, help='Max number of neighbors.')
    parser.add_argument('--drop_prob', type=float, default=0., help="Keep probability for dropout.")
    parser.add_argument('--temp', type=float, default=0.45, help="Softmax temperature")
    # parser.add_argument('--temp', type=float, default=0.05, help="Softmax temperature")
    parser.add_argument('--n_head', type=int, default=4, help="attention head number")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=101, help="Iteration number.")


    parser.add_argument('--eval_step', type=int, default=10, help='How often do an eval step')
    parser.add_argument('--save_rate', type=float, default=0.1, help='How often do save an eval example')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    return parser.parse_args()


DATASETS = [
    # DatasetInfo(name="tr", neigh=100, relevant_neigh=3),
    DatasetInfo(name="simple_dynamic", neigh=4, relevant_neigh=3),
    # DatasetInfo(name="simple", neigh=100, relevant_neigh=4),
    # DatasetInfo(name="simple", neigh=1000, relevant_neigh=4),
    # DatasetInfo(name="simple", neigh=3000, relevant_neigh=4)
]

if __name__ == "__main__":
    args = __pars_args__()
    device = torch.device("cuda:{}".format(args.device) if args.use_cuda else "cpu")
    for dataset in DATASETS:
        print("\n\n---------------")
        print("{}".format(dataset))
        prefix = "{}_neigh-{}_rel-{}".format(dataset.name, dataset.neigh, dataset.relevant_neigh)
        args.dataset_prefix = prefix

        args.max_neighbors = dataset.neigh
        input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor = get_embeddings(path.join("data", args.data_dir),
                                                                                                             prefix=prefix)
        train_dataset = CDataset(path.join("data", args.data_dir), "{}_{}".format(prefix, args.train_file_name))
        eval_dataset = CDataset(path.join("data", args.data_dir), "{}_{}".format(prefix, args.eval_file_name))
        test_dataset = CDataset(path.join("data", args.data_dir), "{}_{}".format(prefix, args.test_file_name))


        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, drop_last=True)

        test_rmse = []
        for exp in range(1):
            print("---------------")
            print("execution nr {}".format(exp+1))
            print("---------------")
            EXP_NAME = "exp-{}_time-{}".format(exp, datetime.now())

            model = JordanRNNJointAttention(args.input_dim, args.hidden_dim, args.output_dim, args.n_head, args.time_windows,
                                      dropout_prob=args.drop_prob,
                                      temperature=args.temp)
            model.name += "_" + prefix
            model = model.to(device)

            model.reset_parameters()
            # model = torch.load(path.join(args.data_dir, "{}.pt".format(model.name)))
            train_fn = setup_model(model, args.batch_size, args, True)
            eval_fn = setup_model(model, args.eval_batch_size, args, False)


            total_loss = []
            eval_number = 0
            eval_loss = []
            best_model = float("infinity")

            for i_iter in range(args.n_iter):
                iter_loss, _ = train_fn(train_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor, device)
                total_loss.append(iter_loss)

                print("{}\t{}".format(i_iter, iter_loss))

                # plot loss
                vis.line(
                    Y=np.array(total_loss),
                    X=np.array(range(i_iter + 1)),
                    opts=dict(
                            # legend=["loss", "penal", "only_loss"],
                            legend=["loss"],
                            title=model.name + " training loos",
                            showlegend=True),
                    win="win:train-{}".format(EXP_NAME))

                if i_iter % args.eval_step == 0:
                    iter_eval, saved_weights = eval_fn(eval_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor, device)
                    eval_loss.append(iter_eval)

                    vis.line(
                        Y=np.array(eval_loss),
                        X=np.array(range(0, i_iter + 1, args.eval_step)),
                        opts=dict(legend=["RMSE"],
                                  title=model.name + " eval loos",
                                  showlegend=True),
                        win="win:eval-{}".format(EXP_NAME))

                    torch.save(saved_weights, ensure_dir(path.join("data", args.data_dir, model.name, "{}_new_saved_eval_iter-{}_temp-{}.bin".format(args.dataset_prefix, int(i_iter/args.eval_step), args.temp))))

                    # pickle.dump(saved_weights, open(ensure_dir(path.join(args.data_dir, model.name, "{}saved_eval_iter-{}_temp-{}.bin".format(args.dataset_prefix, int(i_iter/args.eval_step), args.temp))), "wb"))

                    if best_model > iter_eval:
                        print("save best model")
                        best_model = iter_eval
                        torch.save(model, path.join("data", args.data_dir, "{}.pt".format(model.name)))

            # test performance
            model = torch.load(path.join("data", args.data_dir, "{}.pt".format(model.name)))
            # model = torch.load(path.join("data", args.data_dir, "Jordan_RNN_FeatureTransformerAttention.pt"))
            test_fn = setup_model(model, args.eval_batch_size, args, False)

            iter_test, saved_weights = test_fn(test_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor, device)
            print("test RMSE: {}".format(iter_test))
            test_rmse.append(iter_test)
            torch.save(saved_weights, ensure_dir(path.join("data", args.data_dir, model.name, "{}_new_saved_test_adam_temp-{}.bin".format(args.dataset_prefix, args.temp))))

        print("execution_mean: {}".format(np.mean(test_rmse)))
    # pickle.dump(saved_weights, open(ensure_dir(path.join("data", args.data_dir, model.name, "{}_new_saved_test_adam_temp-{}.bin".format(args.dataset_prefix, args.temp))), "wb"))