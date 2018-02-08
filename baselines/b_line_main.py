from helper import CustomDataset, get_embeddings, RiskToTensor, AttributeToTensor, ensure_dir
from datasets.sintetic.utils import get_sintetic_embeddings

from os.path import join as path_join
from torch.utils.data import DataLoader
from baselines.models import SimpleGRU, StructuralRNN, FeaturedJointSelfAttentionRNN
import torch.optim as optim
from torch.autograd import Variable
import torch
import random

import argparse
import visdom
from datetime import datetime
import pickle

vis = visdom.Visdom()
EXP_NAME = "exp-{}".format(datetime.now())


config = {
  'user': 'root',
  'password': 'vela1990',
  'host': '127.0.0.1',
  'database': 'ml_crif',
}


def __pars_args__():
    parser = argparse.ArgumentParser(description='Guided attention model')
    parser.add_argument("--data_dir", "-d_dir", type=str, default=path_join("..", "data", "sintetic"), help="Directory containing dataset file")
    parser.add_argument("--dataset_prefix", type=str, default="tr_", help="Prefix for the dataset")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_dataset.bin", help="Train file name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str, default="eval_dataset.bin", help="Eval file name")
    parser.add_argument("--test_file_name", "-test_fn", type=str, default="test_dataset.bin", help="Test file name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=30, help='Batch size for eval.')

    parser.add_argument('--input_dim', type=int, default=1, help='Embedding size.')
    parser.add_argument('--hidden_size', type=int, default=5, help='Hidden state memory size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of rnn layers.')
    parser.add_argument('--max_neighbors', "-m_neig", type=int, default=4, help='Max number of neighbors.')
    parser.add_argument('--output_size', type=int, default=1, help='output size.')
    parser.add_argument('--drop_prob', type=float, default=0.0, help="Keep probability for dropout.")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=102, help="Iteration number.")
    parser.add_argument('--eval_step', type=int, default=10, help='How often do an eval step')
    parser.add_argument('--save_rate', type=float, default=0.8, help='How often do save an eval example')
    return parser.parse_args()


def setup_model(model, batch_size, args, is_training=True):
    if is_training:
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        model.eval()
        optimizer = None


    def execute(dataset, input_embeddings, target_embeddings, neighbor_embeddings, seq_len):
        _loss = 0
        saved_weights = {}

        for b_idx, b_index in enumerate(dataset):
            b_input_sequence = Variable(input_embeddings[b_index])
            b_target_sequence = Variable(target_embeddings[b_index])
            b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
            b_seq_len = seq_len[b_index]

            node_hidden = model.init_hidden(batch_size)
            neighbor_hidden = model.init_hidden(args.max_neighbors * batch_size)

            if args.use_cuda:
                b_input_sequence = b_input_sequence.cuda()
                b_target_sequence = b_target_sequence.cuda()
                b_neighbors_sequence = b_neighbors_sequence.cuda()
                b_seq_len = b_seq_len.cuda()

                node_hidden = node_hidden.cuda()
                neighbor_hidden = neighbor_hidden.cuda()

            if is_training:
                optimizer.zero_grad()

            predict, node_hidden, neighbor_hidden = model.forward(b_input_sequence, node_hidden, b_neighbors_sequence, neighbor_hidden,
                                                                  b_seq_len)

            if is_training:
                loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())
            else:
                loss = model.compute_error(predict.squeeze(), b_target_sequence.squeeze())

            _loss += loss.data[0]
            b_idx += 1

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()

                if (b_idx * batch_size) % 1000 == 0:
                    print("num example:{}\tloss:{}".format((b_idx * batch_size), _loss / b_idx))

            elif random.random() > args.save_rate:
                b_input_sequence = b_input_sequence.data.cpu()
                b_target_sequence = b_target_sequence.data.cpu()
                b_neighbors_sequence = b_neighbors_sequence.data.cpu()
                predict = predict.data.cpu().squeeze()
                for row, idx in enumerate(b_index):
                    saved_weights[idx] = dict(
                        id=idx,
                        input=b_input_sequence[row],
                        target=b_target_sequence[row],
                        neighbors=b_neighbors_sequence[row].squeeze(),
                        predict=predict[row]
                    )

        _loss /= b_idx
        return _loss, saved_weights
    return execute

if __name__ == "__main__":
    args = __pars_args__()

    input_embeddings, target_embeddings, neighbor_embeddings, seq_len = get_sintetic_embeddings(args.data_dir, prefix=args.dataset_prefix)
    model = StructuralRNN(args.input_dim, args.hidden_size, args.output_size, args.num_layers, args.max_neighbors, input_embeddings.size(1),
                                                 dropout_prob=args.drop_prob)
    model.reset_parameters()

    train_dataset = CustomDataset(args.data_dir, args.dataset_prefix + args.train_file_name)
    eval_dataset = CustomDataset(args.data_dir, args.dataset_prefix + args.eval_file_name)
    test_dataset = CustomDataset(args.data_dir, args.dataset_prefix + args.test_file_name)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                  drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1,
                                 drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1,
                                 drop_last=True)

    if args.use_cuda:
        model.cuda()

    train = setup_model(model, args.batch_size, args)
    eval = setup_model(model, args.eval_batch_size, args, is_training=False)

    total_loss = []
    eval_number = 0
    eval_loss = []
    best_model = float("infinity")

    for i_iter in range(args.n_iter):
        iter_loss, _ = train(train_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, seq_len)
        total_loss.append(iter_loss)
        print(iter_loss)

        # plot loss
        vis.line(
            Y=torch.FloatTensor(total_loss),
            X=torch.LongTensor(range(i_iter + 1)),
            opts=dict(
                    legend=["loss"],
                    title=model.name + " training loos",
                    showlegend=True),
            win="win:train-{}".format(EXP_NAME))

        if i_iter % args.eval_step == 0:
            iter_eval, saved_weights = eval(eval_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, seq_len)
            eval_loss.append(iter_eval)
            vis.line(
                Y=torch.FloatTensor(eval_loss),
                X=torch.LongTensor(range(0, i_iter + 1, args.eval_step)),
                opts=dict(legend=["RMSE"],
                          title=model.name + " eval loos",
                          showlegend=True),
                win="win:eval-{}".format(EXP_NAME))
            print("dump example")
            pickle.dump(saved_weights, open(ensure_dir(path_join(args.data_dir, model.name, "saved_eval_iter_{}_drop_{}.bin".format(int(i_iter/args.eval_step), args.drop_prob))), "wb"))
            print("dump done")

            if best_model > iter_eval:
                print("save best model")
                best_model = iter_eval
                torch.save(model, path_join(args.data_dir, "model.pt"))


    # test performance
    model = torch.load(path_join(args.data_dir, "model.pt"))
    if model.name == "StructuralRNN":
        model.NodeRNN.flatten_parameters()
        model.NeighborRNN.flatten_parameters()
    elif model.name == "SimpleRNN":
        model.rnn.flatten_parameters()
    else:
        raise NotImplementedError


    test = setup_model(model, args.eval_batch_size, args, is_training=False)
    iter_test, saved_weights = test(test_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, seq_len)
    print("test RMSE: {}",format(iter_test))
    pickle.dump(saved_weights, open(ensure_dir(
        path_join(args.data_dir, model.name, "saved_test_drop_{}.bin".format(args.drop_prob))), "wb"))