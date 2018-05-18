from torch.autograd import Variable
from torch import optim
from torch import nn
import random
from helper import get_time_mask


def setup_model(model, batch_size, args, is_training=True):
    if is_training:
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    def execute(dataset, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neigh):
        _loss = 0
        saved_weights = {}



        for b_idx, b_index in enumerate(dataset):
            b_input_sequence = Variable(input_embeddings[b_index])
            b_target_sequence = Variable(target_embeddings[b_index])
            b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
            b_edge_types = Variable(edge_types[b_index])
            b_mask_neigh = mask_neigh[b_index]
            b_mask_time = get_time_mask(args.time_windows, b_neighbors_sequence.size())

            if args.use_cuda:
                b_input_sequence = b_input_sequence.cuda()
                b_target_sequence = b_target_sequence.cuda()
                b_neighbors_sequence = b_neighbors_sequence.cuda()
                b_edge_types = b_edge_types.cuda()
                b_mask_neigh = b_mask_neigh.cuda()
                b_mask_time = b_mask_time.cuda()

            node_hidden = model.init_hidden(batch_size)
            neighbor_hidden = model.init_hidden(batch_size*args.max_neighbors)

            if is_training:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            predict, neigh_neigh_attention, node_neigh_attention = model.forward(b_input_sequence, b_neighbors_sequence, node_hidden, neighbor_hidden,
                                                                       b_edge_types, b_mask_neigh, b_mask_time, b_target_sequence)

            if is_training:
                loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())
            else:
                loss = model.compute_error(predict.squeeze(), b_target_sequence.squeeze())

            _loss += loss.data[0]
            b_idx += 1

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()

                if (b_idx * args.batch_size) % 1000 == 0:
                    print("num example:{}\tloss:{}".format((b_idx * args.batch_size), _loss / b_idx))

            elif random.random() > args.save_rate:
                b_input_sequence = b_input_sequence.data.cpu()
                b_target_sequence = b_target_sequence.data.cpu()
                b_neighbors_sequence = b_neighbors_sequence.data.cpu()
                b_edge_types = b_edge_types.data.cpu()
                b_mask_neigh = b_mask_neigh.cpu()
                predict = predict.data.cpu().squeeze()
                print(predict)
                # print(b_target_sequence[0], predict[0])
                for row, idx in enumerate(b_index):
                    saved_weights[idx] = dict(
                        id=idx,
                        # neigh_neigh_attention=neigh_neigh_attention[row].cpu(),
                        # node_neigh_attention=node_neigh_attention[row].cpu(),
                        input=b_input_sequence[row],
                        target=b_target_sequence[row],
                        neighbors=b_neighbors_sequence[row].squeeze(),
                        edge_type=b_edge_types[row].squeeze(),
                        mask_neigh=b_mask_neigh[row],
                        predict=predict[row]
                    )
        _loss /= b_idx
        return _loss, saved_weights
    return execute



def eval_fn(model, dataloader, args, input_embeddings, target_embeddings, neighbor_embeddings, edge_types):
    # EVAL
    model.eval()
    iter_error = 0
    saved_weights = {}

    for b_idx, b_index in enumerate(dataloader):
        if args.use_cuda:
            b_index = b_index.cuda()

        b_input_sequence = Variable(input_embeddings[b_index])
        b_target_sequence = Variable(target_embeddings[b_index])
        b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
        b_edge_types = Variable(edge_types[b_index], volatile=True)

        if args.use_cuda:
            b_input_sequence = b_input_sequence.cuda()
            b_target_sequence = b_target_sequence.cuda()
            b_neighbors_sequence = b_neighbors_sequence.cuda()
            b_edge_types = b_edge_types.cuda()

        node_hidden = model.init_hidden(args.eval_batch_size)
        neighbor_hidden = model.init_hidden(args.max_neighbors * args.eval_batch_size)

        predict, weights = model.forward(b_input_sequence, b_neighbors_sequence, node_hidden, neighbor_hidden, b_edge_types, b_target_sequence)

        iter_error += model.compute_error(predict.squeeze(), b_target_sequence.squeeze()).data[0]
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

    return iter_error, saved_weights

def train_fn(model, optimizer, dataloader, args, input_embeddings, target_embeddings, neighbor_embeddings, edge_types):
    # TRAIN
    model.train()
    iter_loss = 0
    if edge_types:
        supervised = True
    else:
        supervised = False

    # iter_penal = 0
    for b_idx, b_index in enumerate(dataloader):
        b_input_sequence = Variable(input_embeddings[b_index])
        b_target_sequence = Variable(target_embeddings[b_index])
        b_neighbors_sequence = Variable(neighbor_embeddings[b_index])
        if supervised:
            b_edge_types = Variable(edge_types[b_index])

        if args.use_cuda:
            b_input_sequence = b_input_sequence.cuda()
            b_target_sequence = b_target_sequence.cuda()
            b_neighbors_sequence = b_neighbors_sequence.cuda()
            if supervised:
                b_edge_types = b_edge_types.cuda()

        node_hidden = model.init_hidden(args.batch_size)
        neighbor_hidden = model.init_hidden(args.max_neighbors * args.batch_size)

        optimizer.zero_grad()
        if supervised:
            predict, weights = model.forward(b_input_sequence, b_neighbors_sequence, node_hidden, neighbor_hidden, b_edge_types, b_target_sequence)
        else:
            predict, weights = model.forward(b_input_sequence, b_neighbors_sequence, node_hidden, neighbor_hidden, b_edge_types,
                                             b_target_sequence)

        loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())

        # print(model.attention.layer_norm.gamma.data)
        # print(model.attention.layer_norm.beta.data)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # print(model.attention.layer_norm.gamma.data)
        # print(model.attention.layer_norm.beta.data)

        iter_loss += loss.data[0]
        # iter_penal += penal.data
        b_idx += 1
        if (b_idx * args.batch_size) % 1000 == 0:
            print("num example:{}\tloss:{}".format((b_idx * args.batch_size), iter_loss/b_idx))
    iter_loss /= b_idx
    return iter_loss