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


TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]

RISK_ATTRIBUTE = ["class_scoring_risk", "val_scoring_risk",
                  "class_scoring_ai", "val_scoring_ai",
                  "class_scoring_bi", "val_scoring_bi",
                  "class_scoring_cr", "val_scoring_cr",
                  "class_scoring_sd", "val_scoring_sd",
                  "pre_notching", "class_scoring_pre"]

C_ATTRIBUTE = ["ateco", "b_partner", "birth_date", "cod_uo", "country_code", "customer_kind", "customer_type", "region",
               "sae", "uncollectable_status", "zipcode"]

REF_DATE = "2018-01-01"
DATE_FORMAT = "%Y-%m-%d"

T_risk = namedtuple("T_risk", RISK_ATTRIBUTE)
T_attribute = namedtuple("T_attribute", C_ATTRIBUTE)



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




def ensure_dir(file_path):
    '''
    Used to ensure the creation of a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path
