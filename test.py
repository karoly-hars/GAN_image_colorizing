import torch
from torch.utils.data import DataLoader
from datasets import get_cifar10_data, extract_cifar10_images, Cifar10Dataset
from networks import Generator
from helpers import save_sample
import warnings
warnings.simplefilter("ignore") # sorry. warnings annoye me
import argparse
import os.path as osp


def main(args):    
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
    
    generator = Generator()
    if use_gpu:
        generator.cuda()
    
    # load the weights
    if use_gpu:
        generator.load_state_dict(torch.load(args.load_path))
    else:
        generator.load_state_dict(torch.load(args.load_path, map_location="cpu"))
    
    # run through the dataset and display the first few images of every batch
    for idx, sample in enumerate(data_loader):
        
        img_l, real_img_lab = sample[:,0:1,:,:], sample
        if use_gpu:
            img_l, real_img_lab = img_l.cuda(), real_img_lab.cuda()
        
        fake_img_ab = generator(img_l).detach()
        fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)
        
        print("sample {}/{}".format(idx+1, len(data_loader)+1))
        save_sample(real_img_lab, fake_img_lab, osp.join(args.save_path, "test_sample_{}.png".format(idx)), show=True)          


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Image colorization with GANs")
    parser.add_argument("--data_path", type=str, default="./data", help="Download and extraction path for the dataset")
    parser.add_argument("--load_path", type=str, default="ep200_weigths_gen.pt", help="path to the generator weights. If the file is not in place, the program will attempt download it")
    parser.add_argument("--save_path", type=str, default="./imgs", help="Save and load path for the network weigths")   
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    main(args)