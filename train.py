import os
import os.path as osp
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import get_cifar10_data, extract_cifar10_images, Cifar10Dataset
from networks import Generator, Discriminator, weights_init_normal, adjust_learning_rate
from helpers import ones_target, zeros_target, show_images, print_losses
import warnings
warnings.simplefilter("ignore") # sorry. warnings annoye me
import argparse


def main(args): 
    # download and extract dataset
    get_cifar10_data(args.data_path)
    data_dirs = extract_cifar10_images(args.data_path)
    
    datasets = {}
    datasets["train"] = Cifar10Dataset(root_dir=data_dirs["train"], mirror=True)
    datasets["test"] = Cifar10Dataset(root_dir=data_dirs["test"], mirror=False, random_seed=0) 
    
    for phase in ["train", "test"]:
        print("{} dataset len: {}".format(phase, len(datasets[phase])))
    
    # define dataloaders    
    data_loaders = {}
    data_loaders["train"] = DataLoader(datasets["train"], batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers)
    data_loaders["test"] = DataLoader(datasets["test"], batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)
    
    # set up models 
    global_step = 0
    use_gpu = torch.cuda.is_available()
    print("use_gpu={}".format(use_gpu))
    
    generator = Generator()
    discriminator = Discriminator()
    if use_gpu:
        generator.cuda()
        discriminator.cuda()
        
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)      
        
    # optimizer adam with reduced momentum
    g_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate_g, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate_d, betas=(0.5, 0.999))
    
    # losses
    l1_loss_fn = nn.L1Loss(reduction="mean")
    discriminator_loss_fn = nn.BCELoss(reduction="mean")
    
    # make save dir, if needed
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # load weight if the training is not starting from the beginning
    if args.start_epoch > 0:
        global_step = args.start_epoch * len(data_loaders["train"])
        generator.load_state_dict(torch.load(osp.join(args.save_path,
                                                      "checkpoint_ep{}_gen.pt".format(args.start_epoch-1))))
        discriminator.load_state_dict(torch.load(osp.join(args.save_path,
                                                          "checkpoint_ep{}_disc.pt".format(args.start_epoch-1))))
    
    #  begin training    
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
                img_l, real_img_lab = sample[:,0:1,:,:], sample
                img_l, real_img_lab = img_l.float(), real_img_lab.float()
                if use_gpu:
                    img_l, real_img_lab = img_l.cuda(), real_img_lab.cuda()
                    
                # generate targets
                target_ones = ones_target(real_img_lab.size(0))
                target_zeros = zeros_target(real_img_lab.size(0))
                if use_gpu:
                   target_ones, target_zeros = target_ones.cuda(), target_zeros.cuda()
        
                if phase == "train":
                    # adjust LR
                    global_step += 1
                    adjust_learning_rate(g_optimizer, args.learning_rate_g, global_step)
                    adjust_learning_rate(d_optimizer, args.learning_rate_d, global_step)
                    
                    # reset generator gradients
                    g_optimizer.zero_grad()
                
                # train / inference the generator
                with torch.set_grad_enabled(phase == "train"):
                    fake_img_ab = generator(img_l)
                    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)
                    if use_gpu:
                        fake_img_lab = fake_img_lab.cuda()
                  
                    # adv loss
                    adv_loss = discriminator_loss_fn(discriminator(fake_img_lab), target_ones)
                    # l1 loss
                    l1_loss = l1_loss_fn(real_img_lab[:,1:,:,:], fake_img_ab)
                    # full gen loss
                    full_gen_loss = adv_loss + (args.lambda_ * l1_loss)
                    
                    if phase == "train":
                        full_gen_loss.backward()
                        g_optimizer.step()
                
                epoch_gen_adv_loss += adv_loss.item()
                epoch_gen_l1_loss += l1_loss.item()
                
                if phase == "train":
                    # reset discriminator gradients
                    d_optimizer.zero_grad()
                    
                # train / inference the discriminator
                with torch.set_grad_enabled(phase == "train"):
                    prediction_real = discriminator(real_img_lab)
                    prediction_fake = discriminator(fake_img_lab.detach())
    
                    loss_real = discriminator_loss_fn(prediction_real, target_ones * args.smoothing)  
                    loss_fake = discriminator_loss_fn(prediction_fake, target_zeros)
                    full_disc_loss = loss_real + loss_fake
                    
                    if phase == "train":
                        full_disc_loss.backward()
                        d_optimizer.step()
                        
                epoch_disc_real_loss += loss_real.item()
                epoch_disc_fake_loss += loss_fake.item()
                epoch_disc_real_acc += np.mean(prediction_real.detach().cpu().numpy()>0.5) 
                epoch_disc_fake_acc += np.mean(prediction_fake.detach().cpu().numpy()<=0.5)
                
                # save the first sample for later
                if phase == "test" and idx == 0:
                    sample_real_img_lab = real_img_lab
                    sample_fake_img_lab = fake_img_lab
                
            # display losses    
            print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
                         epoch_disc_real_loss, epoch_disc_fake_loss,
                         epoch_disc_real_acc, epoch_disc_fake_acc, 
                         len(data_loaders[phase]))  
            
            # save after every nth epoch
            if phase == "test" and epoch % args.save_freq == 0:
                # display sample images          
                torch.save(generator.state_dict(), osp.join(args.save_path, "checkpoint_ep{}_gen.pt".format(epoch)))
                torch.save(discriminator.state_dict(), osp.join(args.save_path, "checkpoint_ep{}_disc.pt".format(epoch)))
                show_images(sample_real_img_lab, sample_fake_img_lab, epoch=epoch, save_dir=args.save_path)  
                print("Checkpoint.")                   
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Image colorization with GANs")
    parser.add_argument("--data_path", type=str, default="./data", help="Download and extraction path for the dataset")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Save and load path for the network weigths")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequience during training.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_epoch", type=int, default=0, help="If start_epoch>0, attempts to a load previously saved weigth from the save_path")
    parser.add_argument("--max_epoch", type=int, default=201)
    parser.add_argument("--learning_rate_g", type=float, default=3e-4, help="Base learning rate for the generator")
    parser.add_argument("--learning_rate_d", type=float, default=3e-5, help="Base learning rate for the discriminator")
    parser.add_argument("--smoothing", type=float, default=0.9)
    parser.add_argument("--lambda_", type=int, default=80)
    args = parser.parse_args()
    
    main(args)
