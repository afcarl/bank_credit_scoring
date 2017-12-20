from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import os
from torch.utils.data import Dataset
from collections import namedtuple
import torch
import pickle
from random import randint

TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]

RISK_ATTRIBUTE = [# "segmento",
                  "class_scoring_risk", "val_scoring_risk",
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

def update_or_plot(i_iter):
    if i_iter == 0:
        return None
    else:
        return "append"


def get_max_length(path):
    return len(pickle.load(open(path, "rb")))

CustomerSample = namedtuple('CustomerSample', ['customer_id', 'risk', 'attribute'])
class CustomerDataset(Dataset):
    def __init__(self, base_path, file_name):
        self.customers = self.__extract_sample__(pickle.load(open(os.path.join(base_path, file_name), "rb")))


        self.max_ateco = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("ateco")))
        self.max_b_partner = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("b_partner")))
        self.max_c_kind = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("c_kind")))
        self.max_c_type = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("c_type")))
        self.max_cod_uo = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("cod_uo")))
        self.max_country_code = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("country_code")))
        self.max_region = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("region")))
        self.max_sae = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("sae")))
        self.max_un_status = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("uncollectable_status")))
        self.max_zipcode = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("zipcode")))

    def __extract_sample__(self, customers):
        create_sample = lambda x: CustomerSample(x['customer_id'], [value for timestemp, value in x["risk"].items()], x["attribute"])
        return [create_sample(customer_att) for customer_att in customers]

    def __attribute_one_hot_tensor(self, attribute):

        b_partner_one_hot = torch.zeros((self.max_b_partner))
        b_partner_one_hot[attribute.b_partner] = 1

        c_kind_one_hot = torch.zeros((self.max_c_kind))
        c_kind_one_hot[attribute.customer_kind] = 1

        c_type_one_hot = torch.zeros((self.max_c_type))
        c_type_one_hot[attribute.customer_type] = 1

        cod_uo_one_hot = torch.zeros((self.max_cod_uo))
        cod_uo_one_hot[attribute.cod_uo] = 1

        country_code_one_hot = torch.zeros((self.max_country_code))
        country_code_one_hot[attribute.country_code] = 1

        region_one_hot = torch.zeros((self.max_region))
        region_one_hot[attribute.region] = 1

        sae_one_hot = torch.zeros((self.max_sae))
        sae_one_hot[attribute.sae] = 1

        un_status_one_hot = torch.zeros((self.max_un_status))
        un_status_one_hot[attribute.uncollectable_status] = 1

        return torch.cat((torch.FloatTensor([attribute.ateco, attribute.birth_date, attribute.zipcode]),
                          b_partner_one_hot,
                          c_kind_one_hot,
                          c_type_one_hot,
                          cod_uo_one_hot,
                          country_code_one_hot,
                          region_one_hot,
                          sae_one_hot,
                          un_status_one_hot
                          ))

    def __attribute_tensor(self, attribute):
        return torch.FloatTensor([attribute.ateco, attribute.birth_date, attribute.zipcode, attribute.b_partner,
                                  attribute.customer_kind, attribute.customer_type, attribute.cod_uo,
                                  attribute.country_code, attribute.region, attribute.sae, attribute.uncollectable_status])


    def __risk_tensor__(self, risk):
        return torch.FloatTensor(risk)

    def __len__(self):
        return len(self.customers)

    def __getitem__(self, idx):
        customer_id, risk, attribute = self.customers[idx]
        torch_risk = self.__risk_tensor__(risk[:-1])
        torch_attribute = self.__attribute_tensor(attribute)

        return (torch.cat((torch_risk, torch_attribute.expand(12, -1)), dim=1),
                torch.FloatTensor([risk[-1].val_scoring_risk]))


class TestDataset(Dataset):
    def __init__(self, length=10):
        self.length = length
        self.samples = [self.__generate_sequence__() for _ in range(2000)]


    def __generate_sequence__(self):
        return [[randint(0, 99)] for _ in range(self.length)]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_sequence = torch.FloatTensor(sample)
        target_sequence = input_sequence[-2]

        return (input_sequence, target_sequence)



NameSample = namedtuple('NameSample', ['category', 'line'])
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
        return [NameSample(category, self.__unicodeToAscii__(line)) for line in lines]

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
