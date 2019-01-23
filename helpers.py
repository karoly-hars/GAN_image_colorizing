import torch
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


def show_images(real_img_lab, fake_img_lab, plot_size=14, scale=2.5, epoch=None, save_dir=None, pause=2000):
    batch_size = real_img_lab.size()[0]
    plot_size = min(plot_size, batch_size)

    # create white canvas
    canvas = np.ones((3*32 + 4*6, plot_size*32 + (plot_size+1)*6 , 3))*255                
    
    real_img_lab = real_img_lab.cpu().numpy()
    fake_img_lab = fake_img_lab.cpu().numpy()
    
    for i in range(0, plot_size):
        # postprocess real and fake samples
        real_lab = np.transpose(real_img_lab[i], (1,2,0))
        real_lab = lab_postprocess(real_lab)
        real_rgb = cv2.cvtColor(real_lab, cv2.COLOR_LAB2BGR)
        
        grayscale = np.expand_dims(cv2.cvtColor(real_rgb, cv2.COLOR_BGR2GRAY), 2)
        
        fake_lab = np.transpose(fake_img_lab[i], (1,2,0))
        fake_lab = lab_postprocess(fake_lab)
        fake_rgb = cv2.cvtColor(fake_lab, cv2.COLOR_LAB2BGR)
        
        # paint
        x = (i+1)*6+i*32
        canvas[6:38, x:x+32, :] = real_rgb
        canvas[44:76, x:x+32, :] = np.repeat(grayscale, 3, axis=2)
        canvas[82:114, x:x+32, :] = fake_rgb

    # scale 
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # save if its needed
    if save_dir is not None and epoch is not None:
        cv2.imwrite(osp.join(save_dir,"checkpoint_ep{}_sample.png".format(epoch)), canvas*255)

    # display sample
    cv2.destroyAllWindows()
    cv2.imshow("Display", canvas)
    cv2.waitKey(pause)    