import torch
from datasets import postprocess
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


def save_sample(real_imgs_lab, fake_imgs_lab, save_path, plot_size=14, scale=2.5, show=False):
    batch_size = real_imgs_lab.size()[0]
    plot_size = min(plot_size, batch_size)

    # create white canvas
    canvas = np.ones((3*32 + 4*6, plot_size*32 + (plot_size+1)*6 , 3), dtype=np.uint8)*255
    
    real_imgs_lab = real_imgs_lab.cpu().numpy()
    fake_imgs_lab = fake_imgs_lab.cpu().numpy()
    
    for i in range(0, plot_size):
        # postprocess real and fake samples
        real_bgr = postprocess(real_imgs_lab[i])
        fake_bgr = postprocess(fake_imgs_lab[i])       
        grayscale = np.expand_dims(cv2.cvtColor(real_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY), 2)
        # paint
        x = (i+1)*6+i*32
        canvas[6:38, x:x+32, :] = real_bgr
        canvas[44:76, x:x+32, :] = np.repeat(grayscale, 3, axis=2)
        canvas[82:114, x:x+32, :] = fake_bgr

    # scale 
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # save 
    cv2.imwrite(osp.join(save_path), canvas)
    
    if show:
        cv2.destroyAllWindows()
        cv2.imshow("sample", canvas)
        cv2.waitKey(100)
        