from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import os
from torch.utils.data import Dataset
from torch.autograd import Variable
from collections import namedtuple
import torch
Sample = namedtuple('Sample', ['category', 'line'])


class NameDataset(Dataset):
    def __init__(self, paths):
        self.paths = glob.glob(paths)

        self.all_letters = " " + string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters) + 1  # Plus EOS marker and 0 pad

        self.lines = []
        self.all_categories = []

        for filename in self.paths:
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = self.__readLines__(category, filename)
            self.lines.extend(lines)

        self.n_categories = len(self.all_categories)
        self.max_sequence_length = max(map(len, map(lambda x: x.line, self.lines)))


    def __unicodeToAscii__(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def __readLines__(self, category, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [Sample(category, self.__unicodeToAscii__(line)) for line in lines]

    # One-hot vector for category
    def __category_tensor__(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(self.n_categories)
        tensor[li] = 1
        return tensor.expand(self.max_sequence_length, self.n_categories)

    # One-hot matrix of first to last letters (not including EOS) for input
    def __tensor_line__(self, line):
        input_tensor = torch.zeros(self.max_sequence_length, self.n_letters)
        target_tensor = torch.LongTensor(self.max_sequence_length).zero_()
        for li in range(len(line)):
            input_tensor[li][self.all_letters.find(line[li])] = 1
            if li == (len(line) - 1):
                target_tensor[li] = self.n_letters - 1 # EOS
            else:
                target_tensor[li] = self.all_letters.find(line[li + 1])
        return input_tensor, target_tensor, len(line)-1



    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sample = self.lines[idx]

        # transform
        category, line = sample
        category_one_hot = self.__category_tensor__(category)
        input_sequence, target_sequence, seq_length = self.__tensor_line__(line)

        return dict(input_sequence=torch.cat((input_sequence, category_one_hot), dim=1),
                    target_sequence=target_sequence,
                    seq_length=seq_length)


# def set_pad_collate(max_sequence_length, batch_size, n_letters):
#     def padd_collate(batch):
#         "Puts each data field into a tensor with outer dimension batch size"
#         b_category = torch.stack([d for d in batch['category']], dim=0)
#         b_input_sequence = torch.LongTensor(batch_size, max_sequence_length, n_letters).zero_()
#         b_target_sequence = torch.LongTensor(batch_size, max_sequence_length).zero_()
#
#         [b_input_sequence[idx] = d for idx, d in enumerate(batch['input_sequence'])]
#
#
#
#         error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
#         elem_type = type(batch[0])
#         if torch.is_tensor(batch[0]):
#             out = None
#             if _use_shared_memory:
#                 # If we're in a background process, concatenate directly into a
#                 # shared memory tensor to avoid an extra copy
#                 numel = sum([x.numel() for x in batch])
#                 storage = batch[0].storage()._new_shared(numel)
#                 out = batch[0].new(storage)
#             return torch.stack(batch, 0, out=out)
#         elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#                 and elem_type.__name__ != 'string_':
#             elem = batch[0]
#             if elem_type.__name__ == 'ndarray':
#                 # array of string classes and object
#                 if re.search('[SaUO]', elem.dtype.str) is not None:
#                     raise TypeError(error_msg.format(elem.dtype))
#
#                 return torch.stack([torch.from_numpy(b) for b in batch], 0)
#             if elem.shape == ():  # scalars
#                 py_type = float if elem.dtype.name.startswith('float') else int
#                 return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
#         elif isinstance(batch[0], int):
#             return torch.LongTensor(batch)
#         elif isinstance(batch[0], float):
#             return torch.DoubleTensor(batch)
#         elif isinstance(batch[0], string_classes):
#             return batch
#         elif isinstance(batch[0], collections.Mapping):
#             return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
#         elif isinstance(batch[0], collections.Sequence):
#             transposed = zip(*batch)
#             return [default_collate(samples) for samples in transposed]
#
#         raise TypeError((error_msg.format(type(batch[0]))))
#     return padd_collate

def ensure_dir(file_path):
    '''
    Used to ensure the creation of a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path
