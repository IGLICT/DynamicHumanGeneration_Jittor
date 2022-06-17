# import torch
import jittor as jt

def create_model(opt):
    if opt.model == 'pix2pixHD':
        if opt.use_everybody:
            print("USE EVERYBODY MODEL ... ")
            from .pix2pixHD_new import Pix2PixHD_Avatar, InferenceModel
        else:
            print("USE Ours MODEL ... ")
            # from .pix2pixHD_Avatar_stage1 import Pix2PixHD_Avatar, InferenceModel
            from .pix2pixHD_Avatar_ori import Pix2PixHD_Avatar, InferenceModel
        # from .pix2pixHD_Avatar_TexG import Pix2PixHD_Avatar, InferenceModel
        if opt.isTrain:
            model = Pix2PixHD_Avatar()
        else:
            model = InferenceModel()
    else:
    	# from .ui_model import UIModel
    	# model = UIModel()
        raise NotImplementedError("Not implemented model")
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    # if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
