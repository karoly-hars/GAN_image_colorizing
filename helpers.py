import torch
import matplotlib.pyplot as plt
from datasets import lab_postprocess
import cv2
import numpy as np
import os.path as osp

def ones_target(size):
    data = torch.ones(size, 1)
    return data


def zeros_target(size):
    data = torch.zeros(size, 1)
    return data

  
def print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss, epoch_disc_real_loss, epoch_disc_fake_loss,
                 epoch_disc_real_acc, epoch_disc_fake_acc, data_loader_len): 
  
    print("  Generator: adversarial loss = {:.4f}, L1 loss = {:.4f})"
         .format(epoch_gen_adv_loss / data_loader_len, epoch_gen_l1_loss / data_loader_len))   
    print("  Discriminator: loss = {:.4f}"
          .format((epoch_disc_real_loss + epoch_disc_fake_loss) / (data_loader_len*2)))
    print("                 acc. = {:.4f} (real acc. = {:.4f}, fake acc. = {:.4f})"
          .format((epoch_disc_real_acc + epoch_disc_fake_acc) / (data_loader_len*2),
                  epoch_disc_real_acc / data_loader_len,
                  epoch_disc_fake_acc / data_loader_len))


def show_images(real_img_lab, fake_img_lab, plot_size=14, epoch=None, save_dir=None, pause_len=0.001):
        
    batch_size = real_img_lab.size()[0]
    plot_size = min(plot_size, batch_size)
    
    real_img_lab = real_img_lab.cpu().numpy()
    fake_img_lab = fake_img_lab.cpu().numpy()
    
    fig0 = plt.figure(0, figsize=(plot_size*1, 3.5))
    
    for i in range(0, plot_size):
        real_lab = np.transpose(real_img_lab[i], (1,2,0))
        real_lab = lab_postprocess(real_lab)
        real_rgb = cv2.cvtColor(real_lab, cv2.COLOR_LAB2RGB)
        real_rgb = (real_rgb*255.0).astype(np.uint8)
        
        fake_lab = np.transpose(fake_img_lab[i], (1,2,0))
        fake_lab = lab_postprocess(fake_lab)
        fake_rgb = cv2.cvtColor(fake_lab, cv2.COLOR_LAB2RGB)
        fake_rgb = (fake_rgb*255.0).astype(np.uint8)       
                      
        plt.subplot(3, plot_size, i+1)
        plt.imshow(real_rgb)
        plt.axis('off')
        
        plt.subplot(3, plot_size, (i+1)+plot_size)
        plt.imshow(cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY), cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, plot_size, 2*plot_size+(i+1))
        plt.imshow(fake_rgb)
        plt.axis('off')
    
    plt.subplots_adjust(top=0.85)
    plt.draw()
    plt.pause(pause_len)
    
    if save_dir is not None and epoch is not None:
        fig0.savefig(osp.join(save_dir,"checkpoint_ep{}_sample.png".format(epoch)))
