import argparse

import torch.nn
import torch.utils.data
import torch.optim

from utils import get_datasets_and_generator, parse_cli, generate


def train(args):
    # Define datasets and generator model.
    uniform_dataloader, normal_dataloader, generator = get_datasets_and_generator(args)
    generator = generator.to(device)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traing generator model '
                                                 'or generate samples.')
    args = parse_cli(parser, train_func=train, generate_func=generate)

    cuda = torch.cuda.is_available()
    device = 'cuda:0' if cuda else 'cpu'

    # Call appropriate function (train or generate).
    args.func(args)
