"""Utilities for naive.py and adversarial.py scripts."""

import torch.nn
import torch.utils.data

import datasets


def get_datasets_and_generator(args):
    # Define datasets.
    uniform_dataset = datasets.UniformRVDataset(args.num_samples, args.shape)
    uniform_dataloader = torch.utils.data.DataLoader(uniform_dataset)
    normal_dataset = datasets.NormalRVDataset(args.num_samples, args.shape)
    normal_dataloader = torch.utils.data.DataLoader(normal_dataset)
    # Define generator model (simple fully-connected with ReLUs).
    generator = torch.nn.Sequential(
        torch.nn.Linear(args.shape, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, args.shape)
    )
    return uniform_dataloader, normal_dataloader, generator


def parse_cli(parser, train_func, generate_func):
    parser.add_argument('num_samples', type=int)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--shape', default=1, type=int)
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', default=5, type=int)
    train_parser.add_argument('--learning-rate', default=1E-3, type=float)
    train_parser.set_defaults(func=train_func)
    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate_func)
    args = parser.parse_args()
    return args


