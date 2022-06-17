import numpy as np
import jittor as jt
import os
# from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
import cv2, copy

from PIL import Image
# import torchvision.transforms as transforms
from jittor import transform

def read_texture_to_tensor(texture_path, tex_size):
    texture = Image.open(texture_path).convert('RGB')
    if texture.width // 4 != tex_size:
        texture = texture.resize((tex_size*4, tex_size*6))
    tex_transform = transform.Compose([transform.ToTensor()]) # set to [0,1]
    texture_tensor = tex_transform(texture) # [3, h, w]
    return jt.array(texture_tensor)

def read_bg_to_tensor(bg_path, bg_size):
    bg = Image.open(bg_path).convert('RGB').resize((bg_size, bg_size))
    bg_transform = [transform.ToTensor(), transform.ImageNormalize([0.5,], [0.5,])] # set to [-1,1]
    bg_transform = transform.Compose(bg_transform)
    bg = bg_transform(bg) # [3, bg_h, bg_w]
    # bg = torch.nn.functional.upsample(bg.unsqueeze(0), size=bg_size, mode='bilinear') # [3, h, w]
    return jt.array(bg[np.newaxis,:])

import copy
def tex_im2tensor(tex_im, tex_size):
    '''
    change texture image [3,h,w] to tensor [part_numx3, tex_size, tex_size] 
    '''
    tex_tensor = jt.zeros([24,3,tex_size,tex_size]) # [part_num, 3, tex_size, tex_size]
    for i in range(4):
        for j in range(6):
            tex_tensor[(6*i+j),:,:,:] = tex_im[:, (tex_size*j):(tex_size*j+tex_size),
                                                    (tex_size*i):(tex_size*i+tex_size)]
    tex_tensor = jt.flip(tex_tensor, dim=2) # do vertical flip
    tex_tensor = tex_tensor.view(-1, tex_size, tex_size) # [part_num x 3, tex_size, tex_size]
    return tex_tensor.unsqueeze(0)

def print_grad(grad):
    # grad = grad * 10000
    print("Gradient shape:", grad.shape, ".............")

def split_consecutive_tensor(x):
    '''
        assume x has size [bs*2, c, h, w]
    '''
    bs = x.shape[0] // 2
    return x[:bs], x[bs:]

class Pix2PixHD_Avatar(BaseModel):
    def name(self):
        return 'Pix2PixHD_Avatar'
    
    def init_loss_filter(self):
        if self.opt.use_densepose_loss:
            flags = (True, True, True, True, True, True, True, True, True, True)
        else:
            flags = (True, True, True, True, True, True, False, False, True, True)
        def loss_filter(g_gan, L2, mask, g_vgg, d_real, d_fake, uv, prob, mask_tex, temp):
            return [l for (l,f) in zip((g_gan,L2,mask,g_vgg,d_real,d_fake, uv, prob, mask_tex, temp),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc # used for TexG and TransG

        ##### define networks
        # Generator network
        TexG_input_nc = 72 + opt.num_class*3
        if opt.pose_plus_laplace:
            TexG_input_nc = 81 + opt.num_class*3
            TexG_input_nc = 81 

        # initialize device information
        # self.device = 'cuda:{}'.format(self.gpu_ids[0])

        if opt.isTrain:
            texture_im = read_texture_to_tensor(opt.texture_path, opt.tex_size)
            # (24x3, tex_size, tex_size)
            texture_tensor = tex_im2tensor(texture_im, opt.tex_size) # (1, 24x3, tex_size, tex_size)
            tmp_tensor = jt.zeros_like(texture_tensor).repeat(1,4,1,1) # (1, 24x3x5, tex_size, tex_size)
            texture_tensor = util.catTextureTensor(texture_tensor, tmp_tensor)
            self.texture = texture_tensor
            # self.texture = jt.nn.Parameter(texture_tensor.to(device=self.device)) 
        self.texture.register_hook(print_grad)

        TransG_input_nc = input_nc = 9
        self.TransG = networks.define_G(TransG_input_nc, (opt.num_class*2,opt.num_class+1), opt.ngf_translate, opt.TransG,
                                      opt.n_downsample_translate, opt.n_blocks_translate, opt.n_local_enhancers,
                                      opt.norm, gpu_ids=self.gpu_ids)

        # self.Feature2RGB = networks.FgFeature2RGB().cuda()
        # self.Feature2RGB = networks.define_G(3+15, 3, 64, "global", opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
        #                               opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.Feature2RGB = networks.define_G(9+15, 3, 64, "global", 3, 5, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        if opt.isTrain:
            bg_tensor = read_bg_to_tensor(opt.bg_path, opt.loadSize)
            # self.BG = torch.nn.Parameter(bg_tensor.to(device=self.device)) # (3,h,w)
            self.BG = bg_tensor

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + 3 # pose + generated image
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # load networks
        TEST = not self.isTrain
        if TEST or opt.continue_train:
            pretrained_path = os.path.join(opt.checkpoints_dir,opt.name)
            texture_path = os.path.join(pretrained_path, "%s_texture.npy" % opt.which_epoch)
            texture_tensor = jt.array(np.load(texture_path)).unsqueeze(0)
            # self.texture = torch.nn.Parameter(texture_tensor.to(device=self.device))
            self.texture = texture_tensor

            bg_path = os.path.join(pretrained_path, "%s_bg.jpg" % opt.which_epoch)
            # bg_path = "/apdcephfs/share_1364276/alyssatan/checkpoints/dance15_18Feature_Temporal/fire_new.jpg"
            bg_tensor = read_bg_to_tensor(bg_path, opt.loadSize)
            self.BG = bg_tensor
            # self.BG = torch.nn.Parameter(bg_tensor.to(device=self.device)) # (3,h,w)

            self.load_network(self.TransG, 'TransG', opt.which_epoch, pretrained_path)
            self.load_network(self.Feature2RGB, 'Feature2RGB', opt.which_epoch, pretrained_path)
        else:
            self.load_network(self.TransG, 'TransG', opt.which_epoch_TransG, opt.load_pretrain_TransG)

        if self.isTrain and opt.continue_train:
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)      

        # has been pre-trained

        if self.opt.verbose or True:
            print('---------- Networks initialized -------------')
            print('parameter number of TransG: %s' % sum(p.numel() for p in self.TransG.parameters()))
            # print('parameter number of TexG: %s' % sum(p.numel() for p in self.TexG.parameters()))
            print('parameter number of Feature2RGB: %s' % sum(p.numel() for p in self.Feature2RGB.parameters()))
            print('parameter number of texture: %s' % sum(p.numel() for p in self.texture))
            if self.isTrain:
                print('parameter number of netD: %s' % sum(p.numel() for p in self.netD.parameters()))

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr2

            # define loss functions
            self.loss_filter = self.init_loss_filter()
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan)
            self.criterionFeat = jt.nn.L1Loss()
            # self.criterionL1_mask = torch.nn.L1Loss(reduction='none')
            self.criterionL1_mask = lambda output, target: (output-target).abs()
            self.criterionL2 = jt.nn.MSELoss()
            # self.criterionL2_mask = jt.nn.MSELoss(reduction='none')
            self.criterionL2_mask = lambda output, target: (output-target).sqr()
            self.criterionMask = jt.nn.BCEWithLogitsLoss()
            # self.criterionMask = torch.nn.BCELoss()
            self.criterion_UV = lambda output, target: (output-target).abs()
            self.criterion_Prob = jt.nn.CrossEntropyLoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','L2', 'mask','G_VGG','D_real', 'D_fake', \
                                                'UV_loss', 'Probs_loss', 'mask_human', 'temporal')

            # meshgrid = jt.stack(jt.meshgrid(jt.linspace(-1,1,200), jt.linspace(-1,1,200)), dim=-1)
            # self.embedder, _ = networks.get_embedder(5)
            # self.meshgrid = self.embedder(meshgrid).unsqueeze(0).permute(0,3,1,2)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = jt.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.lr = opt.lr
            self.beta1 = opt.beta1
            # self.FIRST_EPOCH = 13 # for the small amount data
            self.FIRST_EPOCH = 1 # for All data
            if opt.continue_train:
                self.FIRST_EPOCH = 1 # for pretrained

            self.init_optimizer_G(epoch=1)

            # save tenGrid for given size image
            self.backwarp_tenGrid = {}

        # # initialize device information
        # if len(self.gpu_ids) > 0:
        #     self.device = 'cuda:{}'.format(self.gpu_ids[0])
        # else:
        #     self.device = 'cpu'

    def init_optimizer_G(self, epoch, StaticEpoch=3):
        ratio = 1
        if epoch <= StaticEpoch:
            print("The epoch is %d, Static update !" % epoch)
            # ratio = 0.9 ** (epoch)
            self.optimizer_G = jt.optim.Adam([{'params': self.TransG.parameters(), 'lr': self.lr*ratio},
                                                 {'params': self.texture, 'lr': self.lr},
                                                 {'params': self.Feature2RGB.parameters(), 'lr': self.lr*ratio},
                                                 {'params': self.BG, 'lr': self.lr*5}], betas=(self.beta1, 0.999), lr=self.lr)
        else:
            # ratio = 0.9 ** (epoch)
            print("The epoch is %d, decrease TransG !" % epoch)
            # update All
            self.optimizer_G = jt.optim.Adam([{'params': self.TransG.parameters(), 'lr': self.lr*ratio},
                                                 {'params': self.texture, 'lr': self.lr},
                                                 {'params': self.Feature2RGB.parameters(), 'lr': self.lr},
                                                 {'params': self.BG, 'lr': self.lr*5}], betas=(self.beta1, 0.999), lr=self.lr)
        return self.optimizer_G

    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.size()) not in self.backwarp_tenGrid:
            tenHorizontal = jt.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = jt.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            tenGrid = jt.concat([tenHorizontal, tenVertical], dim=1)
            self.backwarp_tenGrid[str(tenFlow.size())] = jt.concat([ tenHorizontal, tenVertical], 1)
        # end
        tenFlow = jt.concat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
        return jt.nn.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

    def cal_conf(self, flow, flow2, Conf_Thresh=1):
        '''
        calculating the confidence (0/1) of each value in flow
        '''
        grid = util.tensorGrid(flow)
        B, _, H, W = flow.size()
        conf = jt.zeros_like(flow[:,:1,:,:])
        F1to2 = grid + flow # each grid represent point coord in original image, value is new coord in new image [B,2,H,W]
        F1to2[:,0,:,:] = F1to2[:,0,:,:]*2/W - 1
        F1to2[:,1,:,:] = F1to2[:,1,:,:]*2/H - 1

        B2to1 = jt.nn.grid_sample(flow2, F1to2.permute(0,2,3,1), mode='bilinear', padding_mode='border') # diff of new coord point to original point
        # F1to2 is point coord, B2to1 is flow (diff)
        conf[jt.pow((flow + B2to1),2).sum(dim=1, keepdims=True) < Conf_Thresh] = 1
        return conf

    # only for single tensor or list of tensors 
    def encode_input(self, origin):
        if isinstance(origin, list):
            encoded = []
            for item in origin:
                encoded.append(self.encode_input(item))
        else:
            encoded = origin if isinstance(origin, jt.Var) else jt.array(origin)
        return encoded

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = jt.concat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.execute(fake_query)
        else:
            return self.netD.execute(input_concat)


    # def forward(self, epoch, texture, Pose, pose, mask, real, real_uv, real_cls, bg, infer=False):
    def execute(self, epoch, texture, Pose, mask, real, real_uv, real_cls, bg, \
        Pose_before, mask_before, real_image_before, real_uv_before, real_cls_before, flow, flow_inv):

        # Encode Inputs
        texture = self.encode_input([texture])
        Pose, mask, real_image = self.encode_input([Pose, mask, real])
        real_uv, real_cls = self.encode_input([real_uv, real_cls])
        bg = self.encode_input(bg)

        Pose_before, mask_before, real_image_before, real_uv_before, real_cls_before = self.encode_input([\
            Pose_before, mask_before, real_image_before, real_uv_before, real_cls_before])
        flow, flow_inv = self.encode_input([flow, flow_inv])

        # merge conseuctive image
        real_images = jt.concat([real_image, real_image_before], dim=0)
        real_uvs = jt.concat([real_uv, real_uv_before], dim=0)
        real_clss = jt.concat([real_cls, real_cls_before], dim=0)
        masks = jt.concat([mask, mask_before], dim=0)

        mask_tex = 0

        TransG_input = jt.concat([Pose, Pose_before], dim=0)
        UVs, Probs = self.TransG(TransG_input)

        gen_texture = self.texture

        fg_image = util.texture2image(gen_texture, UVs, Probs)
        fg_image_raw = util.texture2image(gen_texture, UVs, Probs, selNUM=3)
        fg_image_raw = fg_image_raw*2 - 1

        # fg_image = self.Feature2RGB(fg_image)
        fg_image = self.Feature2RGB(jt.concat([TransG_input, fg_image], dim=1))

        bg_image = self.BG

        # new bg generation
        norm_Probs = jt.nn.softmax(Probs, dim=1)
        bg_mask = norm_Probs[:,0:1,:,:]
        # bg_mask = 1 - feature_mask # change the mask to the output of FgFeatureRGB
        fake_image = fg_image * (1-bg_mask) + bg_image * (bg_mask)
        fake_image_raw = fg_image_raw * (1-bg_mask) + bg_image * (bg_mask)

        StaticEpoch = 10
        # StaticEpoch = 0
        self.StaticEpoch = StaticEpoch

    ### GAN loss ###
        # Fake Detection and Loss
        # pred_fake_pool_raw = self.discriminate(TransG_input, fake_image_raw, use_pool=False)
        # loss_D_fake = self.criterionGAN(pred_fake_pool_raw, False)
        # loss_D_fake = torch.tensor(0, dtype=torch.float, requires_grad=True).to(self.device)*0
        loss_D_fake = jt.array(0, dtype=jt.float32)
        if epoch > StaticEpoch:
            pred_fake_pool = self.discriminate(TransG_input, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            # loss_D_fake *= 0.5

        # Real Detection and Loss
        # loss_D_real = torch.tensor(0, dtype=torch.float, requires_grad=True).to(self.device)*0
        loss_D_real = jt.array(0, dtype=jt.float32)
        if epoch > StaticEpoch:
            pred_real = self.discriminate(TransG_input, real_images)
            loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        # pred_fake_raw = self.netD.forward(torch.cat((TransG_input, fake_image_raw), dim=1))    
        # loss_G_GAN = self.criterionGAN(pred_fake_raw, True)
        loss_G_GAN = 0
        if epoch > StaticEpoch:
            # pred_fake = self.netD.forward(torch.cat((TransG_input, fake_image), dim=1))
            pred_fake = self.netD.execute(jt.concat((TransG_input, fake_image), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            # loss_G_GAN *= 0.5

    ### VGG loss ###
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.criterionVGG(fake_image_raw, real_images) * self.opt.lambda_feat
            if epoch > StaticEpoch:
                loss_G_VGG += self.criterionVGG(fake_image, real_images) * self.opt.lambda_feat
                # loss_G_VGG *= 0.5

    ### L2 loss ###
        loss_L2_mask = 0
        loss_L2_mask += self.criterionL2_mask(fake_image_raw, real_images) * self.opt.lambda_L2
        if epoch > StaticEpoch:
            loss_L2_mask += self.criterionL2_mask(fake_image, real_images) * self.opt.lambda_L2
            # loss_L2_mask *= 0.5
        # loss_L2_mask = torch.mean(loss_L2_mask)
        loss_L2_mask = jt.mean(loss_L2_mask)
    
    ### UV loss ###
        loss_UV = 0
        UV_mask=[]
        for part_id in range(24):
            UV_mask.append(real_clss==(part_id+1))
            UV_mask.append(real_clss==(part_id+1))
        # UV_mask_tensor=torch.stack(UV_mask,dim=1).float()
        UV_mask_tensor=jt.stack(UV_mask,dim=1).float()
        loss_UV = self.criterion_UV(UVs, real_uvs) * UV_mask_tensor * self.opt.lambda_UV
    ### classify loss ###
        loss_Prob = self.criterion_Prob(Probs, real_clss.long()) * self.opt.lambda_Prob
    ### mask loss ###
        loss_G_mask = 0
         # foreground mask (mask is set to [0,1])
        # loss_mask = self.criterionMask(1-bg_mask, mask) * self.opt.lambda_mask
        loss_mask = self.criterionMask(1-bg_mask, masks) * self.opt.lambda_mask/10
        # loss_mask = self.criterionMask(1-bg_mask, torch.ones_like(mask)) * self.opt.lambda_mask/10 # for dance29 debug
    ### TV loss ###
        # loss_TV = util.All_TVloss(UVs, Probs)
        loss_TV = 0

    ### mask loss ###
        loss_mask_human = 0
        if epoch > StaticEpoch:
            # loss_mask_human = self.criterionL2_mask(fg_image, -1*torch.ones_like(fg_image)) * bg_mask * self.opt.lambda_L2
            loss_mask_human += self.criterionL2_mask(fake_image, real_images) * (1-bg_mask) * self.opt.lambda_L2
            # loss_mask_human *= 0.5
            loss_mask_human = jt.mean(loss_mask_human)

    ### Temporal loss ###
        loss_T = 0
        fake_image_bf = warped_image = warped_real_image = warped_image_comp = conf = jt.zeros_like(fake_image)
        if epoch > StaticEpoch:
            conf = self.cal_conf(flow_inv, flow)
            fake_image_now, fake_image_bf = split_consecutive_tensor(fake_image)
            warped_image = self.backwarp(fake_image_bf, flow_inv)
            warped_real_image = self.backwarp(real_image_before, flow_inv)
            warped_image_comp = warped_image * conf
            loss_T = self.criterionL1_mask(warped_image, fake_image_now) * conf * self.opt.lambda_Temp
            loss_T = jt.mean(loss_T)

#         if self.opt.use_mask_tex_loss:
#             loss_mask_tex = self.criterionMask(mask_tex, real_mask_tex) * self.opt.lambda_mask_tex
        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_L2_mask, loss_mask, loss_G_VGG, loss_D_real, loss_D_fake, loss_UV, loss_Prob, loss_mask_human, loss_T), \
            fg_image, fg_image_raw, bg_image, bg_mask, fake_image, util.pickbaseTexture(gen_texture), UVs, Probs, mask_tex, \
                fake_image_bf, warped_image, warped_real_image, warped_image_comp, conf]

    def inference(self, texture, Pose, bg):
        # Encode Inputs
        # texture, Pose, pose, bg = self.encode_input([texture, Pose, pose, bg])
        texture, Pose, bg = self.encode_input([texture, Pose, bg])

        with jt.no_grad():
            mask_tex = 0
            gen_texture = self.texture
            UVs, Probs = self.TransG(Pose) # [bs,48,h,w], [bs,25,h,w]

            fg_image = util.texture2image(gen_texture, UVs, Probs) # [bs, feature, h, w]

            fg_image_raw = util.texture2image(gen_texture, UVs, Probs, selNUM=3)
            fg_image_raw = fg_image_raw*2 - 1

            # fg_image = self.Feature2RGB(fg_image)
            fg_image = self.Feature2RGB(jt.concat([Pose, fg_image], dim=1))

            bg_image = self.BG
            Probs = jt.nn.softmax(Probs, dim=1)
            bg_mask = Probs[:,0:1,:,:]

            fake_image = fg_image * (1-bg_mask) + bg_image * (bg_mask)
            fake_image_raw = fg_image_raw * (1-bg_mask) + bg_image * (bg_mask)

        # return fg_image, bg_image, bg_mask, fg_image_raw*(1-bg_mask) + (1)*(bg_mask), gen_texture, UVs, Probs, mask_tex
        return fg_image, bg_image, bg_mask, fake_image, fake_image_raw, gen_texture, UVs, Probs, mask_tex
        # return fg_image, bg_image, bg_mask, fake_image_raw, gen_texture, UVs, Probs, mask_tex
        # return fg_image, bg_image, bg_mask, fg_image_raw, gen_texture, UVs, Probs, mask_tex


    def save(self, which_epoch):
        # self.save_network(self.TexG, 'TexG', which_epoch, self.gpu_ids)
        self.save_network(self.TransG, 'TransG', which_epoch, self.gpu_ids)
        self.save_network(self.Feature2RGB, 'Feature2RGB', which_epoch, self.gpu_ids)
        # self.save_network(self.BGnet, 'BGnet', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the TransG generator for a number of iterations, also start finetuning it
        params = list(self.TexG.parameters())
        params += list(self.TransG.parameters())
        params += list(self.BGnet.parameters()) 
        self.optimizer_G = jt.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_all_params(self):
        params = list(self.TransG.parameters())
        params += list(self.BGnet.parameters())
        params += [self.texture]
        print("learning rate : %s ... " % self.opt.lr)
        self.optimizer_G = jt.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def update_texture_params(self):
        # update TransG and BGnet parameters 
        params = [self.texture]
        params += list(self.BGnet.parameters())
        lr = self.opt.lr * 10
        print("learning rate : %s ... " % lr)
        self.optimizer_G = jt.optim.Adam(params, lr=lr, betas=(self.opt.beta1, 0.999))

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHD_Avatar):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

