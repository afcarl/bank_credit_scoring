from helper import CustomerDataset, get_embeddings, RiskToTensor, AttributeToTensor
from simple_baseline.simpleModels import LinearCombinationMean
import argparse
from os.path import join as path_join
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import visdom
from datetime import datetime


vis = visdom.Visdom()
EXP_NAME = "exp-{}".format(datetime.now())


def __pars_args__():
    parser = argparse.ArgumentParser(description='Simple GRU')
    parser.add_argument("--data_dir", "-d_dir",type=str, default=path_join("..", "data", "customers"), help="Directory containing dataset file")
    parser.add_argument("--customer_file_name", "-c_fn", type=str, default="customers_formatted_attribute_risk.bin",
                        help="Customer attirbute file name")
    parser.add_argument("--neighbors_file_name", "-n_fn", type=str, default="customeridx_to_neighborsidx.bin",
                        help="Customer attirbute file name")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument("--eval_file_name", "-eval_fn", type=str,
                        default="eval_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument("--customer_neighbors_file", "-c_to_n_fn", type=str, default="customeridx_to_neighborsidx.bin",
                        help="File name")
    parser.add_argument('--feature_size', type=int, default=2, help='Feature size.')
    parser.add_argument('--embedding_dim', type=int, default=24, help='Embedding size.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate (default: 0.001)')
    parser.add_argument('--max_grad_norm', type=float, default=30.0, help="Clip gradients to this norm.")
    parser.add_argument('--n_iter', type=int, default=131, help="Iteration number.")
    parser.add_argument("--use_cuda", "-cuda", type=bool, default=False, help="Use cuda computation")

    return parser.parse_args()


if __name__ == "__main__":
    args = __pars_args__()
    risk_tsfm = RiskToTensor(args.data_dir)
    attribute_tsfm = AttributeToTensor(args.data_dir)
    input_embeddings, target_embeddings, neighbor_embeddings, seq_len = get_embeddings(args.data_dir,
                                                                                       args.customer_file_name,
                                                                                       args.neighbors_file_name,
                                                                                       args.embedding_dim,
                                                                                       risk_tsfm, attribute_tsfm)

    # customer_id_2_customer_idx = pickle.load(open("../data/customers/customerid_to_idx.bin", "rb"))
    # customer_idx_2_neighbors_idx = pickle.load(open("../data/customers/customeridx_to_neighborsidx.bin", "rb"))

    train_dataset = CustomerDataset(args.data_dir, args.train_file_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    eval_dataset = CustomerDataset(args.data_dir, args.eval_file_name)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                  drop_last=True)
    model = LinearCombinationMean(args.feature_size)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.use_cuda:
        model.cuda()

    eval_number = 0
    total_loss = torch.FloatTensor()
    eval_loss = torch.FloatTensor()

    for i_iter in range(args.n_iter):
        iter_loss = 0
        model.train()

        for idx, b_index in enumerate(train_dataloader):
            b_input_sequence = Variable(input_embeddings[b_index])
            b_target_sequence = Variable(target_embeddings[b_index])
            b_neighbor_embeddings = Variable(neighbor_embeddings[b_index])
            b_seq_len = Variable(seq_len[b_index])

            if args.use_cuda:
                b_input_sequence = b_input_sequence.cuda()
                b_target_sequence = b_target_sequence.cuda()
                b_neighbor_embeddings = b_neighbor_embeddings.cuda()
                b_seq_len = b_seq_len.cuda()

            optimizer.zero_grad()
            b_prediction = model.forward(b_input_sequence, b_neighbor_embeddings, b_seq_len)
            loss = model.compute_loss(b_prediction.squeeze(), b_target_sequence)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()

            iter_loss += loss

        iter_loss /= idx
        print(iter_loss.data)

        if args.use_cuda:
            total_loss = torch.cat((total_loss, iter_loss.data.cpu()))
        else:
            total_loss = torch.cat((total_loss, iter_loss.data))

        # plot loss
        vis.line(
            Y=total_loss,
            X=torch.LongTensor(range(i_iter + 1)),
            opts=dict(legend=["loss"],
                      title="linear combination  training loss {}".format(EXP_NAME),
                      showlegend=True),
            win="win:train-{}".format(EXP_NAME))

        # EVAL
        if i_iter % 10 == 0 and i_iter > 0:
            eval_number += 1
            performance = 0

            model.eval()
            for idx, b_index in enumerate(eval_dataloader):
                b_input_sequence = Variable(input_embeddings[b_index])
                b_target_sequence = Variable(target_embeddings[b_index])
                b_neighbor_embeddings = Variable(neighbor_embeddings[b_index])
                b_seq_len = Variable(seq_len[b_index])


                if args.use_cuda:
                    b_input_sequence = b_input_sequence.cuda()
                    b_target_sequence = b_target_sequence.cuda()
                    b_neighbor_embeddings = b_neighbor_embeddings.cuda()
                    b_seq_len = b_seq_len.cuda()

                b_prediction = model.forward(b_input_sequence, b_neighbor_embeddings, b_seq_len)

                performance += model.compute_error(b_prediction.squeeze(), b_target_sequence)

            performance /= idx

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