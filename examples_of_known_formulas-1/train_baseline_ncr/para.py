import argparse


class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='Band gap')

        # Global parameters
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--end_epoch', type=int, default=100)

        # Data parameters
        self.parser.add_argument('--save_dir', type=str, default='checkpoints/')
        self.parser.add_argument('--results_dir', type=str, default='results/')
        self.parser.add_argument('--data_root', type=str, default='../datasets/new-feynman-i.12.4/')

        # Model parameters
        self.parser.add_argument('--layer_num', type=int, default=9)

        args, _ = self.parser.parse_known_args()

        return args
