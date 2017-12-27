from helper import CustomerDataset, get_embeddings, RiskToTensor, AttributeToTensor
from torch.utils.data import DataLoader
from simple_baseline.simpleModels import SimpleMean
import argparse
from os.path import join as path_join

def __pars_args__():
    parser = argparse.ArgumentParser(description='Simple GRU')
    parser.add_argument("--data_dir", "-d_dir",type=str, default=path_join("..", "data", "customers"), help="Directory containing dataset file")
    parser.add_argument("--customer_file_name", "-c_fn", type=str, default="customers_formatted_attribute_risk.bin",
                        help="Customer attirbute file name")
    parser.add_argument("--train_file_name", "-train_fn", type=str, default="train_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument("--eval_file_name", "-eval_fn", type=str,
                        default="eval_customers_formatted_attribute_risk.bin",
                        help="File name")
    parser.add_argument("--customer_neighbors_file", "-c_to_n_fn", type=str, default="customeridx_to_neighborsidx.bin",
                        help="File name")
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size.')
    return parser.parse_args()


if __name__ == "__main__":
    args = __pars_args__()
    risk_tsfm = RiskToTensor(args.data_dir)
    attribute_tsfm = AttributeToTensor(args.data_dir)
    input_embeddings, target_embeddings = get_embeddings(args.data_dir, args.customer_file_name, args.feature_size,
                                                         risk_tsfm, attribute_tsfm)

    eval_dataset = CustomerDataset(args.data_dir, args.eval_file_name)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                  drop_last=True)
    model = SimpleMean()
    error = 0
    for row, b_index in enumerate(eval_dataloader):
        b_input_sequence = input_embeddings[b_index]
        b_target_sequence = target_embeddings[b_index]

        b_prediction = model.forward(b_input_sequence)
        b_error = model.compute_error(b_prediction, b_target_sequence)
        error += b_error
    error /= row
    print(error)