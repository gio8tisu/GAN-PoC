import argparse

import torch.nn
import torch.utils.data
import torch.optim

import datasets


def train(args):
    # Define loss criterion (MSE).
    criterion = torch.nn.MSELoss()

    # Define optimizer (Adam).
    optimizer = torch.optim.Adam(generator.parameters(), lr=1E-3)

    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        for input_, target in zip(uniform_dataloader, normal_dataloader):
            # Model forward pass.
            output = generator(input_.float())
            # Estimate loss.
            loss = criterion(output, target.float())
            loss.backward()
            # Optimization iteration.
            optimizer.step()
            optimizer.zero_grad()

            print('Loss:', loss.item())

    print('Saving model.')
    torch.save(generator.state_dict(), args.model_path)


def generate(args):
    generator.load_state_dict(torch.load(args.model_path))
    for input_ in uniform_dataloader:
        # Model forward pass.
        output = generator(input_.float())
        print(output.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traing generator model '
                                                 'or generate samples.')
    parser.add_argument('num_samples', type=int)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--shape', default=1, type=int)
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', default=5, type=int)
    train_parser.set_defaults(func=train)
    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = 'cuda:0' if cuda else 'cpu'

    # Define datasets.
    uniform_dataset = datasets.UniformRVDataset(args.num_samples, args.shape)
    uniform_dataloader = torch.utils.data.DataLoader(uniform_dataset)
    normal_dataset = datasets.NormalRVDataset(args.num_samples, args.shape)
    normal_dataloader = torch.utils.data.DataLoader(normal_dataset)

    # Define model (simple fully-connected with ReLUs).
    generator = torch.nn.Sequential(
        torch.nn.Linear(args.shape, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, args.shape)
    ).to(device)

    # Call appropriate function.
    args.func(args)
