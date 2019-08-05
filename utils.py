"""Utilities for naive.py and adversarial.py scripts."""

import torch.nn
import torch.utils.data

import datasets


def get_datasets_and_generator(args, no_target=False):
    """Return random variable dataloaders and generator NN.

    A dataloader is used to iterate through the latent space and
    the target distribution. The latent space dataloader is a
    uniform distribution sampler and the target dataloader is a
    normal distribution sampler.
    If no_target is True, just return the latent space dataloader.
    """
    # Define datasets.
    uniform_dataset = datasets.UniformRVDataset(args.num_samples, args.shape)
    uniform_dataloader = torch.utils.data.DataLoader(uniform_dataset, batch_size=args.batch_size)
    # Define generator model (simple fully-connected with ReLUs).
    generator = torch.nn.Sequential(
        torch.nn.Linear(args.shape, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, args.shape),
        torch.nn.Tanh()
    )
    if no_target:
        return uniform_dataloader, generator
    else:
        normal_dataset = datasets.NormalRVDataset(args.num_samples, args.shape,
                                                  static_sample=args.static_sample)
        normal_dataloader = torch.utils.data.DataLoader(normal_dataset, batch_size=args.batch_size)
        return uniform_dataloader, normal_dataloader, generator


def parse_cli(parser, train_func, generate_func):
    parser.add_argument('num_samples', type=int)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--shape', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', default=5, type=int)
    train_parser.add_argument('--static-sample', action='store_true')
    train_parser.add_argument('--learning-rate', default=1E-3, type=float)
    train_parser.set_defaults(func=train_func)
    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate_func)
    args = parser.parse_args()
    return args


def generate(args):
    """Print samples using saved generator NN."""
    # Define latent space dataset and generator model.
    uniform_dataloader, generator = get_datasets_and_generator(args, no_target=True)
    # generator.to(device)

    generator.load_state_dict(torch.load(args.model_path))
    for input_ in uniform_dataloader:
        # Model forward pass.
        output = generator(input_.float())
        print(output.item())
