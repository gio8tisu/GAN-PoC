import argparse

import torch.nn
import torch.utils.data
import torch.optim

import datasets


def train(args):
    # Define loss criterion (binary cross-entropy).
    criterion = torch.nn.BCELoss()
    # Define optimizer for generator and discriminator (Adam).
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    real_label = 1
    fake_label = 0
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        for input_, target in zip(uniform_dataloader, normal_dataloader):
            input_, target = input_.float().to(device), target.float().to(device)
            # Train discriminator with real batch
            optimizerD.zero_grad()
            # Format batch
            # label = torch.full((input_.size(0),), real_label, device=device)
            label = torch.full_like(input_[:, 0], real_label, device=device)
            # Forward pass real batch through D
            output = discriminator(target).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake batch
            # Generate fake image batch with G
            fake = generator(input_)
            label.fill_(fake_label)
            # Forward pass fake batch through D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            print('Discriminator loss:', errD.item())
            print('Generator loss:', errG.item())

    print('Saving model.')
    torch.save(generator.state_dict(), args.model_path)


def generate(args):
    generator.load_state_dict(torch.load(args.model_path))
    for input_ in uniform_dataloader:
        # Model forward pass.
        output = generator(input_.float())
        print(output.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train generator with adversarial training '
                                                 'or generate samples')
    parser.add_argument('num_samples', type=int)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--shape', default=1, type=int)
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', default=5, type=int)
    train_parser.add_argument('--learning-rate', default=1E-3, type=float)
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

    # Define generator model (simple fully-connected with ReLUs).
    generator = torch.nn.Sequential(
        torch.nn.Linear(args.shape, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, args.shape)
    ).to(device)
    # Define discriminator model (simple fully-connected with ReLUs).
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(args.shape, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid()
    ).to(device)

    # Call appropriate function.
    args.func(args)
