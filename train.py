import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Cifar10Dataset
from networks import Generator, Discriminator, weights_init_normal
from helpers import print_args, print_losses
from helpers import save_sample, adjust_learning_rate


def init_training(args):
    """Initalize the data loader, the networks, the optimizers and the loss functions."""
    datasets = Cifar10Dataset.get_datasets_from_scratch(args.data_path)
    for phase in ["train", "test"]:
        print("{} dataset len: {}".format(phase, len(datasets[phase])))

    # define loaders
    data_loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # set up models
    generator = Generator(args.gen_norm).to(device)
    discriminator = Discriminator(args.disc_norm).to(device)

    # initialize weights
    if args.apply_weight_init:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # optimizer adam with reduced momentum
    optimizers = {
        "gen": torch.optim.Adam(generator.parameters(), lr=args.base_lr_gen, betas=(0.5, 0.999)),
        "disc": torch.optim.Adam(discriminator.parameters(), lr=args.base_lr_disc, betas=(0.5, 0.999))
    }

    # losses
    losses = {
        "l1": torch.nn.L1Loss(reduction="mean"),
        'disc': torch.nn.BCELoss(reduction="mean")
    }

    # make save dir, if needed
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # load weights if the training is not starting from the beginning
    global_step = args.start_epoch * len(data_loaders["train"]) if args.start_epoch > 0 else 0
    if args.start_epoch > 0:

        generator.load_state_dict(torch.load(
            os.path.join(args.save_path, "checkpoint_ep{}_gen.pt".format(args.start_epoch - 1)),
            map_location=device
        ))
        discriminator.load_state_dict(torch.load(
            os.path.join(args.save_path, "checkpoint_ep{}_disc.pt".format(args.start_epoch - 1)),
            map_location=device
        ))

    return global_step, device, data_loaders, generator, discriminator, optimizers, losses


def run_training(args):
    """Initialize and run the training process."""
    global_step, device, data_loaders, generator, discriminator, optimizers, losses = init_training(args)
    #  run training process
    for epoch in range(args.start_epoch, args.max_epoch):
        print("\n========== EPOCH {} ==========".format(epoch))

        for phase in ["train", "test"]:

            # running losses for generator
            epoch_gen_adv_loss = 0.0
            epoch_gen_l1_loss = 0.0

            # running losses for discriminator
            epoch_disc_real_loss = 0.0
            epoch_disc_fake_loss = 0.0
            epoch_disc_real_acc = 0.0
            epoch_disc_fake_acc = 0.0

            if phase == "train":
                print("TRAINING:")
            else:
                print("VALIDATION:")

            for idx, sample in enumerate(data_loaders[phase]):

                # get data
                img_l, real_img_lab = sample[:, 0:1, :, :].float().to(device), sample.float().to(device)

                # generate targets
                target_ones = torch.ones(real_img_lab.size(0), 1).to(device)
                target_zeros = torch.zeros(real_img_lab.size(0), 1).to(device)

                if phase == "train":
                    # adjust LR
                    global_step += 1
                    adjust_learning_rate(optimizers["gen"], global_step, base_lr=args.base_lr_gen,
                                         lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)
                    adjust_learning_rate(optimizers["disc"], global_step, base_lr=args.base_lr_disc,
                                         lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)

                    # reset generator gradients
                    optimizers["gen"].zero_grad()

                # train / inference the generator
                with torch.set_grad_enabled(phase == "train"):
                    fake_img_ab = generator(img_l)
                    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1).to(device)

                    # adv loss
                    adv_loss = losses["disc"](discriminator(fake_img_lab), target_ones)
                    # l1 loss
                    l1_loss = losses["l1"](real_img_lab[:, 1:, :, :], fake_img_ab)
                    # full gen loss
                    full_gen_loss = (1.0 - args.l1_weight) * adv_loss + (args.l1_weight * l1_loss)

                    if phase == "train":
                        full_gen_loss.backward()
                        optimizers["gen"].step()

                epoch_gen_adv_loss += adv_loss.item()
                epoch_gen_l1_loss += l1_loss.item()

                if phase == "train":
                    # reset discriminator gradients
                    optimizers["disc"].zero_grad()

                # train / inference the discriminator
                with torch.set_grad_enabled(phase == "train"):
                    prediction_real = discriminator(real_img_lab)
                    prediction_fake = discriminator(fake_img_lab.detach())

                    loss_real = losses["disc"](prediction_real, target_ones * args.smoothing)
                    loss_fake = losses["disc"](prediction_fake, target_zeros)
                    full_disc_loss = loss_real + loss_fake

                    if phase == "train":
                        full_disc_loss.backward()
                        optimizers["disc"].step()

                epoch_disc_real_loss += loss_real.item()
                epoch_disc_fake_loss += loss_fake.item()
                epoch_disc_real_acc += np.mean(prediction_real.detach().cpu().numpy() > 0.5)
                epoch_disc_fake_acc += np.mean(prediction_fake.detach().cpu().numpy() <= 0.5)

                # save the first sample for later
                if phase == "test" and idx == 0:
                    sample_real_img_lab = real_img_lab
                    sample_fake_img_lab = fake_img_lab

            # display losses    
            print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
                         epoch_disc_real_loss, epoch_disc_fake_loss,
                         epoch_disc_real_acc, epoch_disc_fake_acc,
                         len(data_loaders[phase]), args.l1_weight)

            # save after every nth epoch
            if phase == "test":
                if epoch % args.save_freq == 0 or epoch == args.max_epoch - 1:
                    gen_path = os.path.join(args.save_path, "checkpoint_ep{}_gen.pt".format(epoch))
                    disc_path = os.path.join(args.save_path, "checkpoint_ep{}_disc.pt".format(epoch))
                    torch.save(generator.state_dict(), gen_path)
                    torch.save(discriminator.state_dict(), disc_path)
                    print("Checkpoint.")

                # display sample images
                save_sample(
                    sample_real_img_lab,
                    sample_fake_img_lab,
                    os.path.join(args.save_path, "sample_ep{}.png".format(epoch))
                )


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Image colorization with GANs")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Download and extraction path for the dataset")
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                        help="Save and load path for the network weights")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency during training.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="If start_epoch>0, attempts to a load previously saved weigth from the save_path")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--smoothing", type=float, default=0.9)
    parser.add_argument("--l1_weight", type=float, default=0.99)
    parser.add_argument("--base_lr_gen", type=float, default=3e-4, help="Base learning rate for the generator")
    parser.add_argument("--base_lr_disc", type=float, default=6e-5, help="Base learning rate for the discriminator")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Learning rate decay rate for both networks")
    parser.add_argument("--lr_decay_steps", type=float, default=6e4, help="Learning rate decay steps for both networks")
    parser.add_argument("--gen_norm", type=str, default="batch", choices=["batch", "instance"],
                        help="Defines the type of normalization used in the generator")
    parser.add_argument("--disc_norm", type=str, default="batch", choices=["batch", "instance", "spectral"],
                        help="Defines the type of normalization used in the discriminator")
    parser.add_argument("--apply_weight_init", type=bool, default=True,
                        help="If set to 1, applies the 'weights_init_normal' function from networks.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    # display arguments
    print_args(args)

    run_training(args)
