import torch
from torch.utils.data import DataLoader
from datasets import get_cifar10_data, extract_cifar10_images, Cifar10Dataset
from networks import Generator
from helpers import save_test_sample, print_args
import warnings
warnings.simplefilter("ignore") # sorry. warnings annoye me
import argparse
import os
import os.path as osp


def main(args):    
    # print args
    print_args(args)    
    
    # download and extract dataset
    get_cifar10_data(args.data_path)
    data_dirs = extract_cifar10_images(args.data_path)
    
    dataset = Cifar10Dataset(root_dir=data_dirs["test"], mirror=False, random_seed=1) 
    
    print("test dataset len: {}".format(len(dataset)))
    
    # define dataloader    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # set up model
    use_gpu = torch.cuda.is_available()
    print("use_gpu={}".format(use_gpu))
    
    # download the weights for the generators
    if not os.path.exists("batchnorm_ep200_weigths_gen.pt"):
        print("Downloading model weights for generator with BN...")
        os.system("wget https://www.dropbox.com/s/r33ndl969q83gik/batchnorm_ep200_weigths_gen.pt")
    generator_bn = Generator("batch")
    
    
    if not os.path.exists("spectralnorm_ep100_weights_gen.pt"):
        print("Downloading model weights for generator with SN...")
        os.system("wget https://www.dropbox.com/s/tccxduyqp3dj5dg/spectralnorm_ep100_weights_gen.pt")
    generator_sn = Generator("batch")
        
    if use_gpu:
        generator_bn.cuda()
        generator_sn.cuda()
    
    # load the weights
    if use_gpu:
        generator_bn.load_state_dict(torch.load("batchnorm_ep200_weigths_gen.pt"))
        generator_sn.load_state_dict(torch.load("spectralnorm_ep100_weights_gen.pt"))
    else:
        generator_bn.load_state_dict(torch.load("batchnorm_ep200_weigths_gen.pt", map_location="cpu"))
        generator_sn.load_state_dict(torch.load("spectralnorm_ep100_weights_gen.pt", map_location="cpu"))
        
    # make save dir, if needed
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # run through the dataset and display the first few images of every batch
    for idx, sample in enumerate(data_loader):
        
        img_l, real_img_lab = sample[:,0:1,:,:], sample
        if use_gpu:
            img_l, real_img_lab = img_l.cuda(), real_img_lab.cuda()
        
        # generate images with bn model
        fake_img_ab_bn = generator_bn(img_l).detach()
        fake_img_lab_bn = torch.cat([img_l, fake_img_ab_bn], dim=1)
        
        # generate images with sn model
        fake_img_ab_sn = generator_sn(img_l).detach()
        fake_img_lab_sn = torch.cat([img_l, fake_img_ab_sn], dim=1)
        
        print("sample {}/{}".format(idx+1, len(data_loader)+1))
        save_test_sample(real_img_lab, fake_img_lab_bn, fake_img_lab_sn, 
                         osp.join(args.save_path, "test_sample_{}.png".format(idx)), show=True)          


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Image colorization with GANs")
    parser.add_argument("--data_path", type=str, default="./data", help="Download and extraction path for the dataset")
    parser.add_argument("--save_path", type=str, default="./output_imgs", help="Save path for the test imgs")   
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    main(args)
