from torch import optim
from torch import nn
import torch
import random
from helper import get_time_mask


inv_softlog = lambda x: (torch.exp(x) - 1) * 10
inv_softplus = lambda x: torch.log(torch.exp(x) - 1)
inv_softlog_1 = lambda x: torch.exp(x) - 2


def setup_model(model, batch_size, args, is_training=True):
    if is_training:
        # optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    def execute(dataset, input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neigh, device):
        _loss = 0
        saved_weights = {}



        for b_idx, b_index in enumerate(dataset):
            b_input_sequence = input_embeddings[b_index].to(device)
            b_target_sequence = target_embeddings[b_index].to(device)
            b_neighbors_sequence = neighbor_embeddings[b_index].to(device)
            b_edge_types = edge_types[b_index].to(device)
            b_mask_neigh = mask_neigh[b_index].to(device)
            b_mask_time = get_time_mask(args.time_windows, b_neighbors_sequence.size()).to(device)


            node_hidden = model.init_hidden(batch_size).to(device)
            neighbor_hidden = model.init_hidden(batch_size*args.max_neighbors).to(device)

            if is_training:
                model.train()
                model.zero_grad()
                with torch.set_grad_enabled(True):
                    predict, neigh_neigh_attention, node_neigh_attention = model.forward(b_input_sequence,
                                                                                         b_neighbors_sequence, node_hidden,
                                                                                         neighbor_hidden,
                                                                                         b_edge_types, b_mask_neigh,
                                                                                         b_mask_time, b_target_sequence)

                    loss = model.compute_loss(predict.squeeze(), b_target_sequence.squeeze())
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

            else:
                with torch.set_grad_enabled(False):
                    model.eval()
                    predict, neigh_neigh_attention, node_neigh_attention = model.forward(b_input_sequence,
                                                                                         b_neighbors_sequence, node_hidden,
                                                                                         neighbor_hidden,
                                                                                         b_edge_types, b_mask_neigh,
                                                                                         b_mask_time, b_target_sequence)
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

                    if random.random() > args.save_rate:
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

            _loss += loss.item()
            b_idx += 1

            if (b_idx * args.batch_size) % 1000 == 0:
                print("num example:{}\tloss:{}".format((b_idx * args.batch_size), _loss / b_idx))

        _loss /= b_idx
        return _loss, saved_weights
    return execute