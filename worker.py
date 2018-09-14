from torch.autograd import Variable
from torch import optim
from torch import nn, exp, log
import random
from helper import get_time_mask


inv_softlog = lambda x: (exp(x) - 1) * 10
inv_softplus = lambda x: log(exp(x) - 1)
inv_softlog_1 = lambda x: exp(x) - 2


def setup_model(model, batch_size, args, is_training=True):
    if is_training:
        # optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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
                if args.data_dir == "pems":
                    predict = inv_softlog(predict.squeeze())
                    target = inv_softlog(b_target_sequence.squeeze())
                elif args.data_dir == "utility":
                    predict = inv_softplus(predict.squeeze())
                    target = inv_softplus(b_target_sequence.squeeze())
                elif args.data_dir == "customers":
                    predict = inv_softlog_1(predict.squeeze())
                    target = inv_softlog_1(b_target_sequence.squeeze())
                else:
                    predict = predict.squeeze()
                    target = b_target_sequence.squeeze()

                loss = model.compute_error(predict, target)

            _loss += loss.data[0]
            b_idx += 1

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()

                if (b_idx * args.batch_size) % 1000 == 0:
                    print("num example:{}\tloss:{}".format((b_idx * args.batch_size), _loss / b_idx))

            elif random.random() > args.save_rate:
                print(predict)
                for row, idx in enumerate(b_index):
                    if node_neigh_attention is not None:
                        saved_weights[idx] = dict(
                            id=idx,
                            neigh_neigh_attention=neigh_neigh_attention[row].cpu(),
                            node_neigh_attention=node_neigh_attention[row].cpu(),
                            input=b_input_sequence[row].data.cpu(),
                            target=target[row].data.cpu(),
                            neighbors=b_neighbors_sequence[row].data.cpu().squeeze(),
                            edge_type=b_edge_types[row].data.cpu().squeeze(),
                            mask_neigh=b_mask_neigh[row].cpu(),
                            predict=predict[row].data.cpu().squeeze()
                        )
                    else:
                        saved_weights[idx] = dict(
                            id=idx,
                            neigh_neigh_attention=neigh_neigh_attention[row].cpu(),
                            input=b_input_sequence[row].data.cpu(),
                            target=target[row].data.cpu(),
                            neighbors=b_neighbors_sequence[row].data.cpu().squeeze(),
                            edge_type=b_edge_types[row].data.cpu().squeeze(),
                            mask_neigh=b_mask_neigh[row].cpu(),
                            predict=predict[row].data.cpu().squeeze()
                        )
        _loss /= b_idx
        return _loss, saved_weights
    return execute