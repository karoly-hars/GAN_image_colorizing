import os
import cv2
import numpy as np
from datasets import postprocess


def print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss, epoch_disc_real_loss, epoch_disc_fake_loss,
                 epoch_disc_real_acc, epoch_disc_fake_acc, data_loader_len, l1_weight): 
    """Create a display all the losses and accuracies."""
    print('  Generator: adversarial loss = {:.4f}, L1 loss = {:.4f}, full loss = {:.4f}'.format(
        epoch_gen_adv_loss / data_loader_len,
        epoch_gen_l1_loss / data_loader_len,
        (epoch_gen_adv_loss / data_loader_len)*(1.0-l1_weight) + (epoch_gen_l1_loss / data_loader_len)*l1_weight
    ))

    print('  Discriminator: loss = {:.4f}'.format(
        (epoch_disc_real_loss + epoch_disc_fake_loss) / (data_loader_len*2)
    ))

    print('                 acc. = {:.4f} (real acc. = {:.4f}, fake acc. = {:.4f})'.format(
        (epoch_disc_real_acc + epoch_disc_fake_acc) / (data_loader_len*2),
        epoch_disc_real_acc / data_loader_len,
        epoch_disc_fake_acc / data_loader_len
    ))


def save_sample(real_imgs_lab, fake_imgs_lab, save_path, plot_size=20, scale=2.2, show=False):
    """Create a grid of ground truth, grayscale and colorized images and save + display it to the user."""
    batch_size = real_imgs_lab.size()[0]
    plot_size = min(plot_size, batch_size)

    # create white canvas
    canvas = np.ones((3*32 + 4*6, plot_size*32 + (plot_size+1)*6, 3), dtype=np.uint8)*255
    
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
    cv2.imwrite(os.path.join(save_path), canvas)
    
    if show:
        cv2.destroyAllWindows()
        cv2.imshow('sample', canvas)
        cv2.waitKey(10000)


def save_test_sample(real_imgs_lab, fake_imgs_lab1, fake_imgs_lab2, save_path, plot_size=14, scale=1.6, show=False):
    """
    Create a grid of ground truth,
    grayscale and 2 colorized images (from different sources) and save + display it to the user.
    """
    batch_size = real_imgs_lab.size()[0]
    plot_size = min(plot_size, batch_size)

    # create white canvas
    canvas = np.ones((plot_size*32 + (plot_size+1)*6, 4*32 + 5*8, 3), dtype=np.uint8)*255
    
    real_imgs_lab = real_imgs_lab.cpu().numpy()
    fake_imgs_lab1 = fake_imgs_lab1.cpu().numpy()
    fake_imgs_lab2 = fake_imgs_lab2.cpu().numpy()
    
    for i in range(0, plot_size):
        # post-process real and fake samples
        real_bgr = postprocess(real_imgs_lab[i])
        fake_bgr1 = postprocess(fake_imgs_lab1[i])  
        fake_bgr2 = postprocess(fake_imgs_lab2[i])     
        grayscale = np.expand_dims(cv2.cvtColor(real_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY), 2)
        # paint
        x = (i+1)*6+i*32
        canvas[x:x+32, 8:40, :] = real_bgr
        canvas[x:x+32, 48:80, :] = np.repeat(grayscale, 3, axis=2)
        canvas[x:x+32, 88:120, :] = fake_bgr1
        canvas[x:x+32, 128:160, :] = fake_bgr2

    # scale 
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # save 
    cv2.imwrite(os.path.join(save_path), canvas)
    
    if show:
        cv2.destroyAllWindows()
        cv2.imshow('sample', canvas)
        cv2.waitKey(10000)


def print_args(args):
    """Display args."""
    arg_list = str(args)[10:-1].split(',')
    for arg in arg_list:
        print(arg.strip())
    print('')


def adjust_learning_rate(optimizer, global_step, base_lr, lr_decay_rate=0.1, lr_decay_steps=6e4):
    """Adjust the learning rate of the params of an optimizer."""
    lr = base_lr * (lr_decay_rate ** (global_step/lr_decay_steps))
    if lr < 1e-6:
        lr = 1e-6
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
