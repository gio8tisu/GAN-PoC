import argparse
import os.path

import numpy as np
import torch.nn
import torch.utils.data
import torch.optim
import wandb
import tqdm

from utils import get_datasets_and_generator, parse_cli, generate


np.random.seed(0)
torch.manual_seed(0)


def train(args):
    # Define datasets and generator model.
    uniform_dataloader, normal_dataloader, generator = get_datasets_and_generator(args)
    generator = generator.to(device)
    # Define discriminator model (simple fully-connected with LeakyReLUs).
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(args.out_shape, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid()
    ).to(device)

    # Weights and Biases (wandb) stuff.
    wandb.init(project='gan_poc')
    wandb.config.update({'epochs': args.epochs,
                         'batch_size': args.batch_size,
                         'learning_rate': args.learning_rate,
                         'input_shape': args.in_shape,
                         'output_shape': args.out_shape,
                         })
    wandb.watch((generator, discriminator))

    # Define loss criterion (binary cross-entropy).
    criterion = torch.nn.BCELoss()
    # Define optimizer for generator and discriminator (Adam).
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    real_label = 0.9
    fake_label = 0
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        progress_bar = tqdm.tqdm(zip(uniform_dataloader, normal_dataloader),
                                 total=len(uniform_dataloader))
        for input_, target in progress_bar:
            input_, target = input_.float().to(device), target.float().to(device)
            # Train discriminator with real batch
            optimizerD.zero_grad()
            # Format batch
            label = torch.full_like(target[:, 0, 0], real_label, device=device)
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
            optimizerG.zero_grad()
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

            progress = {'Discriminator loss': errD.item(), 'Generator loss': errG.item()}
            progress_bar.set_postfix(progress)
            wandb.log(progress)

    print('Saving model.')
    torch.save(generator.state_dict(), args.model_path)
    torch.save(generator.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train generator with adversarial training '
                                                 'or generate samples')
    args = parse_cli(parser, train_func=train, generate_func=generate)

    cuda = torch.cuda.is_available()
    device = 'cuda:0' if cuda else 'cpu'

    # Call appropriate function (train or generate).
    args.func(args)
