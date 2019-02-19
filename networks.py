import torch.nn.functional as F
import torch
import torch.nn as nn
from spectral_norm import SpectralNorm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0, activation_fn=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()
        model = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size))
        model.append(activation_fn)
        if dropout>0:
            model.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return x

      
class ConvBlockSpectral(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0, activation_fn=nn.LeakyReLU(0.2)):
        super(ConvBlockSpectral, self).__init__()
        if normalize:
            model = [SpectralNorm(nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))]
        else:
            model = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        model.append(activation_fn)
        if dropout>0:
            model.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return x      
      


class TransConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0, activation_fn=nn.ReLU()):
        super(TransConvBlock, self).__init__()
        
        model = [nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size))
        model.append(activation_fn)
        if dropout>0:
            model.append(nn.Dropout(dropout))        
        self.model = nn.Sequential(*model)
        
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  
        return x


# -------------------
# ---- Generator ----
# -------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.down1 = ConvBlock(1, 64, normalize=True, kernel_size=4, stride=1, padding=0, bias=True, dropout=0.0)
        self.down2 = ConvBlock(64, 128, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down3 = ConvBlock(128, 256, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down4 = ConvBlock(256, 512, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down5 = ConvBlock(512, 512, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        
        self.up1 = TransConvBlock(512, 512, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.5)
        self.up2 = TransConvBlock(1024, 256, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.5)
        self.up3 = TransConvBlock(512, 128, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.up4 = TransConvBlock(256, 64, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.final = ConvBlock(128, 2, normalize=False, kernel_size=1, stride=1, padding=0, bias=True, dropout=0.0, activation_fn=nn.Tanh())

        
    def forward(self, x):
        x = F.upsample(x, size=(35, 35), mode='bilinear')
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        u1 = self.up1(d5,d4)
        u2 = self.up2(u1,d3)
        u3 = self.up3(u2,d2)
        u4 = self.up4(u3,d1)    
        
        x = self.final(u4) 
        return x


# -------------------
# -- Discriminator --
# -------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.down1 = ConvBlock(3, 64, normalize=False, kernel_size=4, stride=1, padding=0, bias=True, dropout=0.0)
        self.down2 = ConvBlockSpectral(64, 128, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down3 = ConvBlockSpectral(128, 256, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down4 = ConvBlockSpectral(256, 512, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.final = ConvBlock(512, 1, normalize=False, kernel_size=4, stride=1, padding=0, bias=True, dropout=0.0, activation_fn=nn.Sigmoid())
        
    def forward(self, x):
        x = F.upsample(x, size=(35, 35), mode='bilinear') 
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        x = self.final(d4)
        x = x.view(x.size()[0], -1)
        return x
        
        
def train_gen(img_l, generator, discriminator, g_optimizer, discriminator_loss_fn, l1_loss_fn, l1_weight, use_gpu):
    # generate target
    target_ones = ones_target(real_img_lab.size(0))
    if use_gpu:
        target_ones = target_ones.cuda()
  
    # reset generator gradients
    g_optimizer.zero_grad()
    
    # train the generator
    with torch.set_grad_enabled(True):
        fake_img_ab = generator(img_l)
        fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)
        if use_gpu:
            fake_img_lab = fake_img_lab.cuda()

        # adv loss
        adv_loss = discriminator_loss_fn(discriminator(fake_img_lab), target_ones)
        # l1 loss
        l1_loss = l1_loss_fn(real_img_lab[:,1:,:,:], fake_img_ab)
        # full gen loss
        full_gen_loss = (1.0-l1_weight)*adv_loss + l1_weight*l1_loss

        full_gen_loss.backward()
        g_optimizer.step()

    return adv_loss.item(), l1_loss.item()
  
  
def train_disc(img_l, generator, discriminator, d_optimizer, discriminator_loss_fn, smoothing, use_gpu):
    # generate targets
    target_ones = ones_target(real_img_lab.size(0))
    target_zeros = zeros_target(real_img_lab.size(0))
    if use_gpu:
        target_ones = target_ones.cuda()
        target_zeros = target_zeros.cuda()
  
  
    # generate fake img
    fake_img_ab = generator(img_l)
    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)
  
    # reset discriminator gradients
    d_optimizer.zero_grad()

    # train the discriminator
    with torch.set_grad_enabled(True):
        prediction_real = discriminator(real_img_lab)
        prediction_fake = discriminator(fake_img_lab.detach())

        if smoothing != 1:
            loss_real = discriminator_loss_fn(prediction_real, target_ones * smoothing)  
        else:
            loss_real = discriminator_loss_fn(prediction_real, target_ones)
        loss_fake = discriminator_loss_fn(prediction_fake, target_zeros)
        full_disc_loss = loss_real + loss_fake

        full_disc_loss.backward()
        d_optimizer.step()

    return loss_real.item(), loss_fake.item(), np.mean(prediction_real.detach().cpu().numpy()>0.5), np.mean(prediction_fake.detach().cpu().numpy()<=0.5)
    
    
def test_gen(img_l, generator, discriminator, discriminator_loss_fn, l1_loss_fn, l1_weight, use_gpu):
    # generate target
    target_ones = ones_target(real_img_lab.size(0))
    if use_gpu:
        target_ones = target_ones.cuda()
    
    # inference the generator
    fake_img_ab = generator(img_l)
    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)
    if use_gpu:
        fake_img_lab = fake_img_lab.cuda()

    # adv loss
    adv_loss = discriminator_loss_fn(discriminator(fake_img_lab), target_ones)
    # l1 loss
    l1_loss = l1_loss_fn(real_img_lab[:,1:,:,:], fake_img_ab)
    # full gen loss
    full_gen_loss = (1.0-l1_weight)*adv_loss + l1_weight*l1_loss

    return adv_loss.item(), l1_loss.item(), fake_img_lab.detach()
    
    
def test_disc(img_l, generator, discriminator, discriminator_loss_fn, smoothing, use_gpu):
    # generate targets
    target_ones = ones_target(real_img_lab.size(0))
    target_zeros = zeros_target(real_img_lab.size(0))
    if use_gpu:
        target_ones = target_ones.cuda()
        target_zeros = target_zeros.cuda()
  
    # generate fake img
    fake_img_ab = generator(img_l)
    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)

    # train the discriminator
    prediction_real = discriminator(real_img_lab)
    prediction_fake = discriminator(fake_img_lab.detach())

    if smoothing != 1:
        loss_real = discriminator_loss_fn(prediction_real, target_ones * smoothing)  
    else:
        loss_real = discriminator_loss_fn(prediction_real, target_ones)
    loss_fake = discriminator_loss_fn(prediction_fake, target_zeros)
    full_disc_loss = loss_real + loss_fake

    return loss_real.item(), loss_fake.item(), np.mean(prediction_real.detach().cpu().numpy()>0.5), np.mean(prediction_fake.detach().cpu().numpy()<=0.5)
  
  
def adjust_learning_rate(optimizer, base_lr, global_step, lr_decay_rate=0.1, lr_decay_steps=6e4):
    lr = base_lr * (lr_decay_rate ** (global_step/lr_decay_steps))
    if lr < 1e-6:
        lr = 1e-6
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
