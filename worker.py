from torch.autograd import Variable
from torch import nn
import random


def eval_fn(model, dataloader, args, input_embeddings, target_embeddings,neighbor_embeddings, seq_len):
    # EVAL
    model.eval()
    iter_error = 0
    iter_norm = 0
    saved_weights = {}

    for b_idx, b_index in enumerate(dataloader):
        if args.use_cuda:
            b_index = b_index.cuda()

        b_input_sequence = Variable(input_embeddings[b_index])
        b_target_sequence = Variable(target_embeddings[b_index])
        b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
        b_seq_len = seq_len[b_index]

        node_hidden = model.init_hidden(args.eval_batch_size)
        neighbor_hidden = model.init_hidden(args.max_neighbors * args.eval_batch_size)

        predict, weights, norm = model.forward(b_input_sequence, b_neighbors_sequence, b_seq_len, node_hidden, neighbor_hidden, b_target_sequence)

        iter_error += model.compute_error(predict.squeeze(), b_target_sequence.squeeze()).data[0]
        iter_norm += norm
        if random.random() > args.save_rate:
            b_input_sequence = b_input_sequence.data.cpu()
            b_target_sequence = b_target_sequence.data.cpu()
            b_neighbors_sequence = b_neighbors_sequence.data.cpu()
            predict = predict.data.cpu().squeeze()
            print(predict)
            for row, idx in enumerate(b_index):
                saved_weights[idx] = dict(
                    id=idx,
                    weights=weights[row].cpu(),
                    input=b_input_sequence[row],
                    target=b_target_sequence[row],
                    neighbors=b_neighbors_sequence[row].squeeze(),
                    predict=predict[row]
                )
    b_idx += 1
    iter_error /= b_idx
    iter_norm /= b_idx

    return iter_error, iter_norm, saved_weights

def train_fn(model, optimizer, dataloader, args, input_embeddings, target_embeddings, neighbor_embeddings, seq_len):
    # TRAIN
    model.train()
    iter_loss = 0
    iter_norm = 0
    # iter_penal = 0
    for b_idx, b_index in enumerate(dataloader):
        if args.use_cuda:
            b_index = b_index.cuda()

        b_input_sequence = Variable(input_embeddings[b_index])
        b_target_sequence = Variable(target_embeddings[b_index])
        b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
        b_seq_len = seq_len[b_index]

        node_hidden = model.init_hidden(args.batch_size)
        neighbor_hidden = model.init_hidden(args.max_neighbors * args.batch_size)

        optimizer.zero_grad()
        predict, weights, norm = model.forward(b_input_sequence, b_neighbors_sequence, b_seq_len, node_hidden, neighbor_hidden, b_target_sequence)
        loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()

        iter_loss += loss.data[0]
        iter_norm += norm
        # iter_penal += penal.data
        b_idx += 1
        if (b_idx * args.batch_size) % 1000 == 0:
            print("num example:{}\tloss:{}".format((b_idx * args.batch_size), iter_loss/b_idx))
    iter_loss /= b_idx
    iter_norm /= b_idx
    return iter_loss, iter_norm