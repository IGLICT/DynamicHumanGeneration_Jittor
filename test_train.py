import time
import os
import numpy as np
import jittor as jt
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import random, cv2

jt.flags.use_cuda = 1

### ignore warning
import warnings
warnings.filterwarnings("ignore")

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 2
    opt.print_freq = 1
    opt.niter = 5
    opt.niter_decay = 0
    opt.max_dataset_size = 10
    opt.niter_fix_global = 0

data_loader = CreateDataLoader(opt)
train_dataset, val_dataset = data_loader.load_data()
print('#training images = %d' % len(train_dataset))
print('#validation images = %d' % len(val_dataset))
dataset_size = int(len(data_loader)*data_loader.ratio)
val_dataset_size = int(len(data_loader)*(1-data_loader.ratio))


### save all code to log ###
import os
tgtdir = os.path.join(opt.checkpoints_dir, opt.name, "code")
def mkdir(tgtdir):
    if not os.path.exists(tgtdir):
        os.mkdir(tgtdir)
        # print("making %s" % tgtdir)

pyfiles = []
for root, dirs, files in os.walk(".", topdown=True):
    curdir = os.path.join(tgtdir, root.strip('./').strip('.'))
    mkdir(curdir)
    for subdir in dirs:
        curdir = os.path.join(curdir, subdir)
        mkdir(curdir)
    for name in files:
        if name.endswith('.py') or name.endswith('.sh'):
            pyfiles.append(os.path.join(root, name))

import shutil
for pyfile in pyfiles:
    tgtfile = os.path.join(tgtdir, pyfile.strip('./'))
    cmd = "cp %s %s" % (pyfile, tgtfile)
    # print(cmd)
    os.system(cmd)
##################################################

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter
valid_step = (start_epoch-1) * val_dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

optimizer_D = model.optimizer_D

### Record the metric for each epoch
csv_path = os.path.join(opt.checkpoints_dir, opt.name, "metric.txt")
csv_lines = []

best_psnr, best_ssim = -np.Inf, -np.Inf

# model.module.Feature2RGB.train()
# model.module.TransG.train()
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    # optimizer_G = model.module.init_optimizer_G(epoch=epoch)

    # if (id(optimizer_G) != id(model.module.optimizer_G)):
    #     print("optimizer_G is not the same with model.optimizer, something is wrong !!!")
    #     input()

    train_fake_imgs, train_real_imgs = [], []
    val_fake_imgs, val_real_imgs = [], []

    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(train_dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        print("processing the %d data" % i)
        for k, v in data.items():
            if isinstance(v, jt.Var):
                print(k, v.shape)
        print("....................")

        ############## Forward Pass ######################

        # t1 = time.time()
        # losses, fg_image, fg_image_raw, bg_image, bg_mask, fake_image, gen_texture, UVs, Probs, mask_tex, fake_image_before, warped_image, warped_real_image, warped_image_comp, conf =\
        #                                 model(epoch, data['texture'], data['Pose'], \
        #                                 data['mask'], data['real'], data['pc'], data['pa'], data['bg'], \
        #                                 data['Pose_before'], data['mask_before'], data['real_before'], data['pc_before'], data['pa_before'], \
        #                                 data['flow'], data['flow_inv'])
