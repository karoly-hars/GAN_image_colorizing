import torch.nn.functional as F
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0, activation_fn=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()
        model = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size, 0.8))
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
            model.append(nn.BatchNorm2d(out_size, 0.8))
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
        self.down2 = ConvBlock(64, 128, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down3 = ConvBlock(128, 256, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.down4 = ConvBlock(256, 512, normalize=True, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        
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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def adjust_learning_rate(optimizer, base_lr, global_step, lr_decay_rate=0.1, lr_decay_steps=1e4):
    lr = base_lr * (lr_decay_rate ** (global_step/lr_decay_steps))
    if lr < 1e-6:
        lr = 1e-6
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr