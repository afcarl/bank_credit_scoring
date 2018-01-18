from helper import CustomerDataset, get_embeddings, RiskToTensor, AttributeToTensor, ensure_dir
from datasets.sintetic.utils import get_sintetic_embeddings

from os.path import join as path_join
from torch.utils.data import DataLoader
from models import SimpleStructuredNeighborAttentionRNN
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
    parser.add_argument("--data_dir", "-d_dir", type=str, default=path_join("data", "sintetic"), help="Directory containing dataset file")
    parser.add_argument("--dataset_prefix", type=str, default="sin_", help="Prefix for the dataset")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="sin_train_dataset.bin", help="Train file name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str, default="sin_eval_dataset.bin", help="Eval file name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='Batch size for eval.')

    parser.add_argument('--input_dim', type=int, default=1, help='Embedding size.')
    parser.add_argument('--hidden_size', type=int, default=1, help='Hidden state memory size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of rnn layers.')
    parser.add_argument('--attention_dim', type=int, default=100, help='Attention dim.')
    parser.add_argument('--attention_heads', type=int, default=2, help='Attention heads.')
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



def eval(model, dataloader, input_embeddings, target_embeddings,neighbor_embeddings, seq_len, save_rate):
    # EVAL
    performance = 0
    model.eval()
    node_hidden = model.init_hidden(args.eval_batch_size)
    neighbor_hidden = model.init_hidden(args.max_neighbors * args.eval_batch_size)
    saved_weights = {}
    for b_idx, b_index in enumerate(dataloader):
        b_input_sequence = Variable(input_embeddings[b_index])
        b_target_sequence = Variable(target_embeddings[b_index])
        b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
        b_seq_len = seq_len[b_index]

        node_hidden = model.repackage_hidden_state(node_hidden)
        neighbor_hidden = model.repackage_hidden_state(neighbor_hidden)

        if args.use_cuda:
            b_input_sequence = b_input_sequence.cuda()
            b_target_sequence = b_target_sequence.cuda()
            b_neighbors_sequence = b_neighbors_sequence.cuda()
            b_seq_len = b_seq_len.cuda()

            node_hidden = node_hidden.cuda()
            neighbor_hidden = neighbor_hidden.cuda()

        predict, node_hidden, neighbor_hidden, weights = model.forward(b_input_sequence, node_hidden, b_neighbors_sequence, neighbor_hidden,
                                                                  b_seq_len)
        performance += model.compute_error(predict.squeeze(), b_target_sequence.squeeze())
        if random.random() > save_rate:
            b_input_sequence = b_input_sequence.data.cpu()
            b_target_sequence = b_target_sequence.data.cpu()
            b_neighbors_sequence = b_neighbors_sequence.data.cpu()
            predict = predict.data.cpu().squeeze()
            for row, idx in enumerate(b_index):
                half = "upper" if b_input_sequence[row, -1, 0] >= 5 else "lower"
                saved_weights[idx] = dict(
                    id=idx,
                    weights=weights[row],
                    input=b_input_sequence[row].t(),
                    half=half,
                    target=b_target_sequence[row],
                    neighbors=b_neighbors_sequence[row].squeeze(),
                    predict=predict[row]
                )

    performance = performance.data/(b_idx + 1)
    return performance, saved_weights

def train(model, optimizer, dataloader, input_embeddings, target_embeddings,neighbor_embeddings, seq_len):
    # TRAIN
    model.train()
    node_hidden = model.init_hidden(args.batch_size)
    neighbor_hidden = model.init_hidden(args.max_neighbors * args.batch_size)
    iter_loss = 0
    # iter_penal = 0
    for b_idx, b_index in enumerate(dataloader):
        b_input_sequence = Variable(input_embeddings[b_index])
        b_target_sequence = Variable(target_embeddings[b_index])
        b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
        b_seq_len = seq_len[b_index]

        node_hidden = model.repackage_hidden_state(node_hidden)
        neighbor_hidden = model.repackage_hidden_state(neighbor_hidden)

        if args.use_cuda:
            b_input_sequence = b_input_sequence.cuda()
            b_target_sequence = b_target_sequence.cuda()
            b_neighbors_sequence = b_neighbors_sequence.cuda()
            b_seq_len = b_seq_len.cuda()

            node_hidden = node_hidden.cuda()
            neighbor_hidden = neighbor_hidden.cuda()

        optimizer.zero_grad()

        predict, node_hidden, neighbor_hidden, weights = model.forward(b_input_sequence, node_hidden, b_neighbors_sequence, neighbor_hidden,
                                                                       b_seq_len)
        loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()

        iter_loss += loss.data
        # iter_penal += penal.data
        b_idx += 1
        if (b_idx * args.batch_size) % 1000 == 0:
            print("num example:{}\tloss:{}".format((b_idx * args.batch_size), iter_loss[0]/b_idx))
    iter_loss /= b_idx
    # iter_penal /= b_idx
    # return iter_loss, iter_penal
    return iter_loss


if __name__ == "__main__":
    args = __pars_args__()

    # if args.use_cuda:
    #     torch.cuda.manual_seed_all(10)
    # else:
    #     torch.cuda.manual_seed(10)


    input_embeddings, target_embeddings, neighbor_embeddings, seq_len = get_sintetic_embeddings(args.data_dir, prefix=args.dataset_prefix)
    model = SimpleStructuredNeighborAttentionRNN(args.input_dim, args.hidden_size, args.output_size, args.num_layers,
                                                 args.max_neighbors, input_embeddings.size(1), args.attention_heads,
                                                 dropout_prob=args.drop_prob)

    train_dataset = CustomerDataset(args.data_dir,  args.train_file_name)
    eval_dataset = CustomerDataset(args.data_dir, args.eval_file_name)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=1,
                                 drop_last=True)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    if args.use_cuda:
        model.cuda()


    total_loss = torch.FloatTensor()
    # total_penal = torch.FloatTensor()
    # total_only_loss = torch.FloatTensor()
    eval_number = 0
    eval_loss = torch.FloatTensor()

    for i_iter in range(args.n_iter):
        iter_loss = train(model, optimizer, train_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, seq_len)
        if args.use_cuda:
            total_loss = torch.cat((total_loss, iter_loss.cpu()))
            # total_penal = torch.cat((total_penal, iter_penal.cpu()))
            # total_only_loss = torch.cat((total_only_loss, (iter_loss - iter_penal).cpu()))
        else:
            total_loss = torch.cat((total_loss, iter_loss))
            # total_penal = torch.cat((total_penal, iter_penal))
            # total_only_loss = torch.cat((total_only_loss, iter_loss-iter_penal))
        print(iter_loss)

        # plot loss
        vis.line(
            # Y=torch.stack((total_loss, total_penal, total_only_loss), dim=0).t(),
            Y=total_loss,
            X=torch.LongTensor(range(i_iter + 1)),
            opts=dict(
                    # legend=["loss", "penal", "only_loss"],
                    legend=["loss"],
                    title=model.name + " training loos",
                    showlegend=True),
            win="win:train-{}".format(EXP_NAME))

        if i_iter % args.eval_step == 0:
            iter_eval, saved_weights = eval(model, eval_dataloader, input_embeddings, target_embeddings, neighbor_embeddings, seq_len, args.save_rate)
            if args.use_cuda:
                eval_loss = torch.cat((eval_loss, iter_eval.cpu()))
            else:
                eval_loss = torch.cat((eval_loss, iter_eval))

            vis.line(
                Y=eval_loss,
                X=torch.LongTensor(range(0, i_iter + 1, args.eval_step)),
                opts=dict(legend=["RMSE"],
                          title=model.name + " eval loos",
                          showlegend=True),
                win="win:eval-{}".format(EXP_NAME))

            pickle.dump(saved_weights, open(ensure_dir(path_join(args.data_dir, model.name, "saved_eval_iter_{}.bin".format(int(i_iter/args.eval_step)))), "wb"))