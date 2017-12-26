from helper import CustomerDataset, accuracy, rmse, TestDataset
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


def __pars_args__():
    parser = argparse.ArgumentParser(description='Simple GRU')
    parser.add_argument("--data_dir", "-d_dir",type=str, default=path_join("..", "data", "customers"), help="Directory containing dataset file")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument("--eval_file_name", "-eval_fn", type=str,
                        default="eval_customers_formatted_attribute_risk.bin",
                        help="File name")

    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")

    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='Batch size for eval.')
    parser.add_argument('--feature_size', type=int, default=184, help='Feature size.')
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
    model = SimpleGRU(args.feature_size, args.memory_size, 2, args.output_size, args.batch_size,
                      dropout=args.drop_prob)

    # train_dataset = TestDataset(10)
    # eval_dataset = TestDataset(10)

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
        hidden = model.init_hidden(args.batch_size)
        # TRAIN
        for i_batch, (b_input_sequence, b_target_sequence) in enumerate(train_dataloader):
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
            for i_batch, (b_input_sequence, b_target_sequence) in enumerate(eval_dataloader):
                b_input_sequence = Variable(b_input_sequence, volatile=True)
                b_target_sequence = Variable(b_target_sequence)
                hidden = model.repackage_hidden_state(hidden)

                if args.use_cuda:
                    b_input_sequence = b_input_sequence.cuda()
                    b_target_sequence = b_target_sequence.cuda()
                    hidden = hidden.cuda()

                predict, hidden = model.forward(b_input_sequence, hidden)

                performance += rmse(predict.squeeze(), b_target_sequence.squeeze())

            performance /= i_batch

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

