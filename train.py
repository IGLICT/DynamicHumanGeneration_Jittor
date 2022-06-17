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

        ############## Forward Pass ######################

        # t1 = time.time()
        losses, fg_image, fg_image_raw, bg_image, bg_mask, fake_image, gen_texture, UVs, Probs, mask_tex, fake_image_before, warped_image, warped_real_image, warped_image_comp, conf =\
                                        model(epoch, data['texture'], data['Pose'], \
                                        data['mask'], data['real'], data['pc'], data['pa'], data['bg'], \
                                        data['Pose_before'], data['mask_before'], data['real_before'], data['pc_before'], data['pa_before'], \
                                        data['flow'], data['flow_inv'])

        # t2 = time.time()
        # print("forward time: ", t2-t1)
        # sum per device losses
        losses = [jt.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        # loss_G = loss_dict['G_GAN'] + loss_dict.get('mask',0) + loss_dict.get('G_VGG',0) + loss_dict.get('L2',0) \
        #             + loss_dict.get('UV_loss',0) + loss_dict.get('Probs_loss',0) + loss_dict.get('mask_human', 0) + loss_dict.get('temporal', 0)
        loss_G = loss_dict['G_GAN'] + loss_dict.get('L2',0) + loss_dict.get('G_VGG',0) + loss_dict.get('UV_loss',0) \
                    + loss_dict.get('Probs_loss',0) + loss_dict.get('mask',0)

        ############### Backward Pass ####################
        # # update generator weights
        # optimizer_G.zero_grad()
        # if opt.fp16:                                
        #     with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        # else:
        #     loss_G.backward()          
        # optimizer_G.step()

        # # update discriminator weights
        # optimizer_D.zero_grad()
        # if opt.fp16:                                
        #     with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        # else:
        #     loss_D.backward()
        # optimizer_D.step()     

        # update generator weights
        model.optimizer_G.zero_grad()
        model.optimizer_G.backward(loss_G)
        # loss_G.backward()
        model.optimizer_G.step()

        # update discriminator weights
        model.optimizer_D.zero_grad()
        model.optimizer_D.backward(loss_D)
        model.optimizer_D.step()
        # t3 = time.time()
        # print("backward time:", t3-t2)

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            visuals = OrderedDict()
            visuals['synthesized_texture'] = util.visualizeTex(gen_texture[0])
            visuals['synthesized_fg'] = util.tensor2im(fg_image[0])
            visuals['synthesized_fg_raw'] = util.tensor2im(fg_image_raw[0])
            visuals['synthesized_bg'] = util.tensor2im(bg_image[0])
            visuals['synthesized_mask'] = util.tensor2im(1-bg_mask[0])
            visuals['synthesized_image'] = util.tensor2im(fake_image[0])
            visuals['synthesized_image_before'] = util.tensor2im(fake_image_before[0])
            visuals['synthesized_warp_image'] = util.tensor2im(warped_image[0])
            visuals['synthesized_warp_comp_image'] = util.tensor2im(warped_image_comp[0])
            visuals['synthesized_warp_real_image'] = util.tensor2im(warped_real_image[0])
            visuals['real_image'] = util.tensor2im(data['real'][0])
            visuals['real_before_image'] = util.tensor2im(data['real_before'][0])
            visuals['pose'] = util.tensor2im(data['Pose'][0])
            im_Probs, im_Probs_GT = util.draw_part_assign(Probs[0], data['pa'][0])
            visuals['Probs'] = im_Probs
            visuals['Probs_GT'] = im_Probs_GT
            im_U, im_V = util.draw_uv_coordinate(UVs[0], Probs[0])
            visuals['U'] = im_U
            visuals['V'] = im_V
            im_U_GT, im_V_GT = util.draw_uv_coordinate(data['pc'][0], data['pa'][0])
            visuals['U_GT'] = im_U_GT
            visuals['V_GT'] = im_V_GT
            visualizer.display_current_results(visuals, epoch, total_steps)
            if opt.display_freq == 1:
                input("generate next ? ...")

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            # np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            print("Current model is %s" % (opt.name))

        if epoch_iter >= dataset_size:
            break

        ### Record generate image for evaluation
        train_fake_imgs.append(util.tensor2im(fake_image[0])) 
        train_real_imgs.append(util.tensor2im(data['real'][0]))  # [list of [h,w,3]]

    psnr, ssim = util.calMetrics(train_fake_imgs[::10], train_real_imgs[::10])
    csv_lines.append("Epoch %d:" % (epoch))
    csv_lines.append("      Train        : PSNR %f,    SSIM %f" % (psnr, ssim))


#     cv2.imwrite('latest_texture.jpg', cur_texture)

    # validation
    if not opt.debug:
        # model.module.Feature2RGB.eval()
        # model.module.TransG.eval()
        for i, data in enumerate(val_dataset, start=epoch_iter):
                valid_step += opt.batchSize
                with jt.no_grad():
                    losses, fg_image, fg_image_raw, bg_image, bg_mask, fake_image, gen_texture, UVs, Probs, mask_tex, fake_image_before, warped_image, warped_real_image, warped_image_comp, conf = \
                                                    model(epoch, data['texture'], data['Pose'], \
                                                    data['mask'], data['real'], data['pc'], data['pa'], data['bg'], \
                                                    data['Pose_before'], data['mask_before'], data['real_before'], data['pc_before'], data['pa_before'], \
                                                    data['flow'], data['flow_inv'])
                # sum per device losses
                losses = [ jt.mean(x) if not isinstance(x, int) else x for x in losses ]
                val_loss_names = ['val'+x for x in model.loss_names]
                loss_dict = dict(zip(val_loss_names, losses))

                if valid_step % val_dataset_size == 0:
                    errors = {k: v.numpy() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                    visualizer.plot_current_errors(errors, valid_step)

                    val_visuals = OrderedDict()
                    val_visuals['val_synthesized_texture'] = util.visualizeTex(gen_texture[0])
                    val_visuals['val_synthesized_image'] = util.tensor2im(fake_image[0])
                    val_visuals['val_synthesized_fg'] = util.tensor2im(fg_image[0])
                    val_visuals['val_real_image'] = util.tensor2im(data['real'][0])
                    val_visuals['val_pose'] = util.tensor2im(data['Pose'][0])
                    im_Probs, im_Probs_GT = util.draw_part_assign(Probs[0], data['pa'][0])
                    val_visuals['val_Probs'] = im_Probs
                    val_visuals['val_Probs_GT'] = im_Probs_GT
                    im_U, im_V = util.draw_uv_coordinate(UVs[0], Probs[0])
                    val_visuals['val_U'] = im_U
                    val_visuals['val_V'] = im_V
                    im_U_GT, im_V_GT = util.draw_uv_coordinate(data['pc'][0], data['pa'][0])
                    val_visuals['val_U_GT'] = im_U_GT
                    val_visuals['val_V_GT'] = im_V_GT

                    visualizer.display_current_results(val_visuals, epoch, valid_step)

                ### Record generate image for evaluation
                val_fake_imgs.append(util.tensor2im(fake_image[0])) 
                val_real_imgs.append(util.tensor2im(data['real'][0]))  # [list of [3,h,w]]

        psnr, ssim = util.calMetrics(val_fake_imgs, val_real_imgs, is_train=False)
        csv_lines.append("      Validation: PSNR %f,    SSIM %f" % (psnr, ssim))

    ### if current epoch performs best in the validation model, record it
    if psnr > best_psnr and ssim > best_ssim:
        best_psnr, best_ssim = psnr, ssim
        print("save the current best epoch ... ")
        model.save("best")
        if not opt.use_everybody:
            cur_texture = model.texture[0].numpy()
            tex_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_texture.npy' % ("best"))
            np.save(tex_path, cur_texture)
            cur_texture = util.visualizeTex(gen_texture[0])
            tex_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_texture.jpg' % ("best"))
            cv2.imwrite(tex_path, cur_texture[:,:,::-1])
            cur_bg = util.tensor2im(bg_image[0])
            bg_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_bg.jpg' % ("best"))
            cv2.imwrite(bg_path, cur_bg[:,:,::-1])
        csv_lines.append(" ********* Record the best model in epoch %s ! ********* " % epoch)

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    util.write_csv(csv_path, csv_lines)

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        # model.module.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### save texture for conveniet
        cur_texture = model.texture[0].numpy()
        tex_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_texture.npy' % (epoch))
        np.save(tex_path, cur_texture)

        cur_texture = util.visualizeTex(gen_texture[0])
        tex_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_texture.jpg' % (epoch))
        cv2.imwrite(tex_path, cur_texture[:,:,::-1])

        ### save bg image for conveniet
        cur_bg = util.tensor2im(bg_image[0])
        bg_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_bg.jpg' % (epoch))
        cv2.imwrite(bg_path, cur_bg[:,:,::-1])

    # ### instead of only training the local enhancer, train the entire network after certain iterations
    # if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
    #     print(" ****** update fixed parameters ! ******")
    #     model.module.update_fixed_params()
    #     optimizer_G = model.module.optimizer_G

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()
