from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
from helper import NameDataset
from os.path import join as path_join





class Net(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout=0.5):
        super(Net, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=True,
                          dropout=dropout)


        self.dense = nn.Sequential(nn.Linear(hidden_size, output_size),
                                   nn.ELU())
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)


        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nlayers = n_layers

        self.reset_parameters()

        self.criterion = nn.NLLLoss()



    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform(p)

    def forward(self, input, hidden):
        packed_output, hidden = self.rnn(input, hidden)
        output, seq_length = pad_packed_sequence(packed_output, batch_first=True)

        # select last output (wrong in this case)
        # row_indices = torch.arange(0, batch_size).long()
        # seq_length = torch.LongTensor(seq_length) - 1
        # output = output[row_indices, seq_length, :]

        output = self.dense(output)
        output = self.drop(output)
        return self.softmax(output)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.nlayers, batch_size, self.hidden_size))

    def compute_loss(self, b_predict, b_target):
        return self.criterion(b_predict, b_target)


name_dataset = NameDataset(path_join("..", "data", "language", "names", "*.txt"))


input_size = name_dataset.n_letters + name_dataset.n_categories
output_size = name_dataset.n_letters
batch_size = 60
n_layers = 1
hidden_size = 128

dataloader = DataLoader(name_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

model = Net(input_size, hidden_size, n_layers, output_size, dropout=0.1)
model.train()

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters


learning_rate = 0.0025
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def traing(b_input_sequence, b_target_sequence, b_seq_length):
    hidden = model.init_hidden(batch_size)

    b_seq_length, perm_idx = b_seq_length.sort(0, descending=True)
    b_input_sequence = b_input_sequence[perm_idx]
    b_target_sequence = b_target_sequence[perm_idx]
    b_target_sequence = Variable(b_target_sequence[:, :b_seq_length[0]].contiguous())

    pack = pack_padded_sequence(Variable(b_input_sequence), b_seq_length.numpy(), batch_first=True)

    optimizer.zero_grad()
    loss = 0

    predict = model(pack, hidden)
    predict = predict.view(-1, name_dataset.n_letters)
    loss += model.compute_loss(predict, b_target_sequence.view(-1))

    loss.backward()
    optimizer.step()

    return predict, loss




for iter in range(1, n_iters + 1):
    for i_batch, batch in enumerate(dataloader):

        output, loss = traing(batch['input_sequence'], batch['target_sequence'], batch['seq_length'])
        total_loss += loss

    print(total_loss)
    total_loss = 0

    # if iter % print_every == 0:
    #     print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    # if iter % plot_every == 0:
    #     all_losses.append(total_loss / plot_every)

