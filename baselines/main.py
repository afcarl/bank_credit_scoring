from helper import CustomerDataset, update_or_plot
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
EXP_NAME = "exp-{}".format(datetime.now())


config = {
  'user': 'root',
  'password': 'vela1990',
  'host': '127.0.0.1',
  'database': 'ml_crif',
}

def __pars_args__():
    parser = argparse.ArgumentParser(description='Simple GRU')
    parser.add_argument("--data_dir", "-d_dir",type=str, default=path_join("..", "data", "customers"), help="Directory containing dataset file")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str,
                        default="eval_customers_formatted_attribute_risk.bin",
                        help="File name")

    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')
    parser.add_argument('--feature_size', type=int, default=200, help='Feature size.')
    parser.add_argument('--memory_size', type=list, default=[1024, 518], help='Hidden state memory size.')
    parser.add_argument('--output_size', type=int, default=1, help='output size.')
    parser.add_argument('--drop_prob', type=float, default=0.1, help="Keep probability for dropout.")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=5000, help="Iteration number.")


    parser.add_argument('--train', default=True, help='if we want to update the master weights')
    return parser.parse_args()


if __name__ == "__main__":
    args = __pars_args__()
    model = SimpleGRU(args.feature_size, args.memory_size, 1, args.output_size, args.batch_size,
                      dropout=args.drop_prob)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = CustomerDataset(args.data_dir,  args.eval_file_name)
    eval_dataset = CustomerDataset(args.data_dir, args.eval_file_name)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)

    eval_number = 0
    for i_iter in range(1, args.n_iter):
        total_loss = 0
        model.train()
        hidden = model.init_hidden()
        # TRAIN
        for i_batch, (b_input_sequence, b_target_sequence) in enumerate(train_dataloader):
            b_input_sequence = Variable(b_input_sequence)
            b_target_sequence = Variable(b_target_sequence)
            hidden = model.repackage_hidden_state(hidden)

            optimizer.zero_grad()
            predict = model.forward(b_input_sequence, hidden)
            loss = model.compute_loss(predict, b_target_sequence)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss

        print(total_loss)
        # plot loss
        vis.line(
            Y=total_loss.data,
            X=torch.LongTensor([i_iter - 1]),
             opts=dict(legend=["loss", ],
                       title="Simple GRU training loss",
                       showlegend=True),
             win="win:train-{}".format(EXP_NAME),
             update=update_or_plot(i_iter - 1))


        # EVAL
        if i_iter % 10 == 0:
            print("eval step")
            model.eval()
            mse = 0
            hidden = model.init_hidden()
            for i_batch, (b_input_sequence, b_target_sequence) in enumerate(eval_dataloader):
                b_input_sequence = Variable(b_input_sequence, volatile=True)
                b_target_sequence = Variable(b_target_sequence)
                predict = model.forward(b_input_sequence, hidden)
                mse += torch.nn.functional.mse_loss(predict, b_target_sequence)

                hidden = model.repackage_hidden_state(hidden)
            vis.line(
                Y=mse.data,
                X=torch.LongTensor([eval_number]),
                opts=dict(legend=["MSE", ],
                          title="Simple GRU eval error",
                          showlegend=True),
                win="win:eval-{}".format(EXP_NAME),
                update=update_or_plot(eval_number))
            eval_number += 1
