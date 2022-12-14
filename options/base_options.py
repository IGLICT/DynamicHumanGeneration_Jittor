import argparse
import os
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/') 
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        # initial texture
        self.parser.add_argument('--texture_path', type=str, default='./datasets/tex.jpg')
        self.parser.add_argument('--bg_path', type=str, default='./datasets/bg.jpg')
        self.parser.add_argument('--mask_path', type=str, default='./datasets/')
        self.parser.add_argument('--densepose_path', type=str, default='./datasets/')
        self.parser.add_argument('--pose_path', type=str, default='./datasets/')
        self.parser.add_argument('--pose_tgt_path', type=str, default='./datasets/')
        self.parser.add_argument('--img_path', type=str, default='./datasets/')
        self.parser.add_argument('--num_class', type=int, default=24, help='number of densepose part')
        self.parser.add_argument('--out_tex_nc', type=int, default=72, help='# of output image channels')
        self.parser.add_argument('--part_texture_path', type=str, default='./datasets/')
        self.parser.add_argument('--pose_texture_path', type=str, default='./datasets/')


        # Texture Generator
        self.parser.add_argument('--TexG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--ngf_global', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--use_mask_texture', action='store_true', help='if specified, generate texture with mask') 
        self.parser.add_argument('--tex_size', type=int, default=200, help='size of each part texture')

        # Texture Generator
        self.parser.add_argument('--TransG', type=str, default='multihead', help='selects model to use for netG')
        self.parser.add_argument('--n_downsample_translate', type=int, default=4, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_translate', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--ngf_translate', type=int, default=64, help='# of gen filters in first conv layer')

        # Background Generator
        self.parser.add_argument('--deep_fill_path', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--BGnet', type=str, default='global', help='selects model to use for BGnet')
        self.parser.add_argument('--n_downsample_bg', type=int, default=3, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_bg', type=int, default=5, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--ngf_bg', type=int, default=32, help='# of gen filters in first conv layer')

        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        
        self.parser.add_argument('--use_laplace', action='store_true', help='if specified, use multi-channel laplace projection as label') 
        self.parser.add_argument('--lapalce_path', type=str, default='./datasets/')

        self.parser.add_argument('--data_ratio', type=float, default=0.99, help='ratio used to train, the other used to validation')
        self.parser.add_argument('--pose_plus_laplace', action='store_true', help='concate pose and laplace (81 channel) as input') 

        self.parser.add_argument('--use_everybody', action='store_true', help='use everybody model')
        self.parser.add_argument('--test_val', action='store_true', help='generate validation image')

        self.initialized = True

    def parse(self, save=True, argsList=None):
        if not self.initialized:
            self.initialize()
        if argsList is None:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(args=argsList)

        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
