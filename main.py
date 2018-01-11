from helper import CustomerDataset, get_embeddings, RiskToTensor, AttributeToTensor
from datasets.sintetic.utils import get_sintetic_embeddings

from os.path import join as path_join
from torch.utils.data import DataLoader
from models import SimpleStructuredNeighborAttentionRNN
import torch.optim as optim
from torch.autograd import Variable
import torch

import argparse
import visdom
from datetime import datetime

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
    parser.add_argument("--data_dir", "-d_dir",type=str, default=path_join("data", "sintetic"), help="Directory containing dataset file")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_dataset.bin", help="Train file name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str, default="eval_dataset.bin", help="Eval file name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch size for eval.')

    parser.add_argument('--input_dim', type=int, default=1, help='Embedding size.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden state memory size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of rnn layers.')
    parser.add_argument('--attention_dim', type=int, default=40, help='Attention dim.')
    parser.add_argument('--attention_hops', type=int, default=20, help='Attention hops.')
    parser.add_argument('--max_neighbors', "-m_neig", type=int, default=4, help='Max number of neighbors.')
    parser.add_argument('--output_size', type=int, default=1, help='output size.')
    parser.add_argument('--drop_prob', type=float, default=0.1, help="Keep probability for dropout.")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=131, help="Iteration number.")


    parser.add_argument('--train', default=True, help='if we want to update the master weights')
    return parser.parse_args()


if __name__ == "__main__":
    args = __pars_args__()
    input_embeddings, target_embeddings, neighbor_embeddings, seq_len = get_sintetic_embeddings(args.data_dir)
    model = SimpleStructuredNeighborAttentionRNN(args.input_dim, args.hidden_size, args.output_size, args.num_layers,
                                                 args.attention_dim, args.attention_hops, args.max_neighbors, input_embeddings.size(1),
                                                 dropout_prob=args.drop_prob)

    train_dataset = CustomerDataset(args.data_dir,  args.train_file_name)
    eval_dataset = CustomerDataset(args.data_dir, args.eval_file_name)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=4,
                                 drop_last=True)


    if args.use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    eval_number = 0
    total_loss = torch.FloatTensor()
    eval_loss = torch.FloatTensor()
    for i_iter in range(args.n_iter):
        iter_loss = 0
        model.train()
        node_hidden = model.init_hidden(1)
        neighbor_hidden = model.init_hidden(args.max_neighbors)
        # TRAIN
        for b_idx, b_index in enumerate(train_dataloader):
            b_input_sequence = Variable(input_embeddings[b_index])
            b_target_sequence = Variable(target_embeddings[b_index])
            b_neighbors_sequence = Variable(neighbor_embeddings[b_index]).squeeze(dim=0)
            b_seq_len = seq_len[b_index]

            node_hidden = model.repackage_hidden_state(node_hidden)
            neighbor_hidden = model.repackage_hidden_state(neighbor_hidden)

            if args.use_cuda:
                b_input_sequence = b_input_sequence.cuda()
                b_target_sequence = b_target_sequence.cuda()
                b_neighbors_sequence = b_neighbors_sequence.cuda()

                node_hidden = node_hidden.cuda()
                neighbor_hidden = neighbor_hidden.cuda()

            optimizer.zero_grad()

            predict, node_hidden, neighbor_hidden = model.forward(b_input_sequence, node_hidden, b_neighbors_sequence, neighbor_hidden, b_seq_len)
            loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()

            iter_loss += loss

        iter_loss /= (b_idx+1)

        if args.use_cuda:
            total_loss = torch.cat((total_loss, iter_loss.data.cpu()))
        else:
            total_loss = torch.cat((total_loss, iter_loss.data))

        print(iter_loss.data)

        # plot loss
        vis.line(
            Y=total_loss,
            X=torch.LongTensor(range(i_iter+1)),
             opts=dict(legend=["loss"],
                       title="Simple GRU training loss {}".format(EXP_NAME),
                       showlegend=True),
             win="win:train-{}".format(EXP_NAME))


        # EVAL
        if i_iter % 10 == 0 and i_iter > 0:
            eval_number += 1
            performance = 0
            
            model.eval()
            hidden = model.init_hidden(args.eval_batch_size)
            for b_idx, b_index in enumerate(eval_dataloader):
                b_input_sequence = Variable(input_embeddings[b_index])
                b_target_sequence = Variable(target_embeddings[b_index])
                hidden = model.repackage_hidden_state(hidden)

                if args.use_cuda:
                    b_input_sequence = b_input_sequence.cuda()
                    b_target_sequence = b_target_sequence.cuda()
                    hidden = hidden.cuda()

                predict, hidden = model.forward(b_input_sequence, hidden)

                performance += model.compute_error(predict.squeeze(), b_target_sequence.squeeze())

            performance /= (b_idx+1)

            if args.use_cuda:
                eval_loss = torch.cat((eval_loss, performance.data.cpu()))
            else:
                eval_loss = torch.cat((eval_loss, performance.data))

            vis.line(
                Y=eval_loss,
                X=torch.LongTensor(range(eval_number)),
                opts=dict(legend=["MSE", ],
                          title="Simple GRU eval error",
                          showlegend=True),
                win="win:eval-{}".format(EXP_NAME))

