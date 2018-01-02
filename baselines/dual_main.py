from helper import CustomerDataset, update_or_plot, TestDataset
from os.path import join as path_join
from torch.utils.data import DataLoader
from baselines.models import SimpleGRU
import torch.optim as optim
from torch.autograd import Variable
import torch

import argparse
import visdom
from datetime import datetime

vis = visdom.Visdom()
vis.ipv6 = False
EXP_NAME = "exp-{}".format(datetime.now())

config = {
    'user': 'root',
    'password': 'vela1990',
    'host': '127.0.0.1',
    'database': 'ml_crif',
}


def accuracy(predict, target):
    correct = (target.eq(predict.round())).sum()
    return correct.float() / predict.size(0)

def train_model(model, dataloader, total_loss, args):
    """
    execute a training step
    """
    model.train()
    hidden = model.init_hidden(args.batch_size)
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    iter_loss = 0
    for i_batch, (b_input_sequence, b_target_sequence) in enumerate(dataloader):
        b_input_sequence, b_target_sequence = Variable(b_input_sequence), Variable(b_target_sequence)
        hidden = model.repackage_hidden_state(hidden)

        if args.use_cuda:
            b_input_sequence = b_input_sequence.cuda()
            b_target_sequence = b_target_sequence.cuda()
            hidden = hidden.cuda()

        optimizer.zero_grad()
        predict, hidden = model.forward(b_input_sequence, hidden)
        loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()
        iter_loss += loss
    iter_loss /= i_batch

    if args.use_cuda:
        return torch.cat((total_loss, iter_loss.data.cpu()))
    else:
        return torch.cat(total_loss, iter_loss.data)

def eval_model(model, dataloader, eval_loss, args):
    model.eval()
    hidden = model.init_hidden(args.eval_batch_size)

    performance = 0
    for i_batch, (b_input_sequence, b_target_sequence) in enumerate(dataloader):
        b_input_sequence = Variable(b_input_sequence, volatile=True)
        b_target_sequence = Variable(b_target_sequence)
        hidden = model.repackage_hidden_state(hidden)

        if args.use_cuda:
            b_input_sequence = b_input_sequence.cuda()
            b_target_sequence = b_target_sequence.cuda()
            hidden = hidden.cuda()

        predict, hidden = model.forward(b_input_sequence, hidden)
        performance += torch.nn.functional.mse_loss(predict.squeeze(), b_target_sequence.squeeze())
    performance /= i_batch

    if args.use_cuda:
        return torch.cat((eval_loss, performance.data.cpu()))
    else:
        return torch.cat((eval_loss, performance.data))


def __pars_args__():
    parser = argparse.ArgumentParser(description='Simple GRU')
    parser.add_argument("--data_dir", "-d_dir", type=str, default=path_join("..", "data", "customers"),
                        help="Directory containing dataset file")
    parser.add_argument("--train_file_name", "-train_fn", type=str,
                        default="train_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str,
                        default="eval_customers_formatted_attribute_risk.bin",
                        help="File name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=True, help="Use cuda computation")

    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='Batch size for eval.')
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size.')
    parser.add_argument('--memory_size', type=list, default=[512, 256], help='Hidden state memory size.')
    parser.add_argument('--output_size', type=int, default=1, help='output size.')
    parser.add_argument('--drop_prob', type=float, default=0.1, help="Keep probability for dropout.")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=131, help="Iteration number.")

    parser.add_argument('--train', default=True, help='if we want to update the master weights')
    return parser.parse_args()


if __name__ == "__main__":
    args = __pars_args__()
    high_model = SimpleGRU(args.feature_size, args.memory_size, 2, args.output_size, args.batch_size,
                           dropout=args.drop_prob)
    low_model = SimpleGRU(args.feature_size, args.memory_size, 2, args.output_size, args.batch_size,
                          dropout=args.drop_prob)

    high_risk_train_dataset = CustomerDataset(args.data_dir, "high_" + args.train_file_name)
    low_risk_train_dataset = CustomerDataset(args.data_dir, "low_" + args.train_file_name)

    high_risk_eval_dataset = CustomerDataset(args.data_dir, "high_" + args.eval_file_name)
    low_risk_eval_dataset = CustomerDataset(args.data_dir, "low_" + args.eval_file_name)

    high_train_dataloader = DataLoader(high_risk_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                       drop_last=True)
    low_train_dataloader = DataLoader(low_risk_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                      drop_last=True)

    high_eval_dataloader = DataLoader(high_risk_eval_dataset, batch_size=args.eval_batch_size, shuffle=True,
                                      num_workers=4, drop_last=True)
    low_eval_dataloader = DataLoader(low_risk_eval_dataset, batch_size=args.eval_batch_size, shuffle=True,
                                     num_workers=4, drop_last=True)

    high_total_loss = torch.FloatTensor()
    low_total_loss = torch.FloatTensor()
    high_eval_loss = torch.FloatTensor()
    low_eval_loss = torch.FloatTensor()
    if args.use_cuda:
        high_model.cuda()
        low_model.cuda()


    eval_number = 0

    for i_iter in range(args.n_iter):
        # TRAIN HIGH
        high_total_loss = train_model(high_model, high_train_dataloader, high_total_loss, args)
        vis.line(
            Y=high_total_loss,
            X=torch.arange(i_iter + 1).long(),
            opts=dict(legend=["loss"],
                      title="Dual GRU HIGH loss {}".format(EXP_NAME),
                      showlegend=True),
            win="win:high_train-{}".format(EXP_NAME))

        # TRAIN LOW
        low_total_loss = train_model(low_model, low_train_dataloader, low_total_loss, args)
        vis.line(
            Y=low_total_loss,
            X=torch.arange(i_iter + 1).long(),
            opts=dict(legend=["loss"],
                      title="Dual GRU LOW loss {}".format(EXP_NAME),
                      showlegend=True),
            win="win:low_train-{}".format(EXP_NAME))




        if i_iter % 10 == 0 and i_iter > 0:
            # EVAL HIGH
            high_eval_loss = eval_model(high_model, high_eval_dataloader, high_eval_loss, args)
            vis.line(
                Y=high_eval_loss,
                X=torch.arange(eval_number + 1).long(),
                opts=dict(legend=["MSE", ],
                          title="Dual GRU HIGH eval error",
                          showlegend=True),
                win="win:high_eval-{}".format(EXP_NAME))

            low_eval_loss = eval_model(low_model, low_eval_dataloader, low_eval_loss, args)
            vis.line(
                Y=low_eval_loss,
                X=torch.arange(eval_number + 1).long(),
                opts=dict(legend=["MSE", ],
                          title="Dual GRU LOW eval error",
                          showlegend=True),
                win="win:low_eval-{}".format(EXP_NAME))

            eval_number += 1

