import os
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import Cifar10Dataset
from networks import Generator
from helpers import save_test_sample, print_args


def init_test(args):
    """Create the data loader, and the generators for testing purposes."""
    # create loader
    dataset = Cifar10Dataset.get_datasets_from_scratch(args.data_path)["test"]
    print("Test dataset len: {}".format(len(dataset)))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # download the weights for the generators
    if not os.path.exists("batchnorm_ep200_weigths_gen.pt"):
        print("Downloading model weights for generator with BN...")
        os.system("wget https://www.dropbox.com/s/r33ndl969q83gik/batchnorm_ep200_weigths_gen.pt")

    if not os.path.exists("spectralnorm_ep100_weights_gen.pt"):
        print("Downloading model weights for generator with SN...")
        os.system("wget https://www.dropbox.com/s/tccxduyqp3dj5dg/spectralnorm_ep100_weights_gen.pt")

    # load generator that was trained with batch norm
    generator_bn = Generator(normalization_type="batch").to(device)
    # load generator that was trained with spectral norm
    generator_sn = Generator(normalization_type="batch").to(device)

    # load the weights
    generator_bn.load_state_dict(torch.load("batchnorm_ep200_weigths_gen.pt", map_location=device))
    generator_sn.load_state_dict(torch.load("spectralnorm_ep100_weights_gen.pt", map_location=device))

    # make save dir, if needed
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return device, data_loader, generator_bn, generator_sn


def run_test(args):
    """Run the networks on the test set, and save/show the samples."""
    device, data_loader, generator_bn, generator_sn = init_test(args)

    # run through the dataset and display the first few images of every batch
    for idx, sample in enumerate(data_loader):

        img_l, real_img_lab = sample[:, 0:1, :, :].to(device), sample.to(device)

        # generate images with bn model
        fake_img_ab_bn = generator_bn(img_l).detach()
        fake_img_lab_bn = torch.cat([img_l, fake_img_ab_bn], dim=1)

        # generate images with sn model
        fake_img_ab_sn = generator_sn(img_l).detach()
        fake_img_lab_sn = torch.cat([img_l, fake_img_ab_sn], dim=1)

        print("sample {}/{}".format(idx + 1, len(data_loader) + 1))
        save_test_sample(real_img_lab, fake_img_lab_bn, fake_img_lab_sn,
                         os.path.join(args.save_path, "test_sample_{}.png".format(idx)), show=True)


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Image colorization with GANs")
    parser.add_argument("--data_path", type=str, default="./data", help="Download and extraction path for the dataset")
    parser.add_argument("--save_path", type=str, default="./output_imgs", help="Save path for the test imgs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()

    # display args
    print_args(args)

    run_test(args)
