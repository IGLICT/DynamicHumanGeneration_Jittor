import glob, os
import imageio
import util.util as util
import cv2
import numpy as np

loadSize = 512

scale, translate = util.scale_resize((512,512), (1024,1024,3), mean_height=0.0)

# kpt_dir = "../../DanceDataset/dance15/openpose_json/*.json"
# kpt_dir = "../../DynamicTexture/code/keypoints/*.json"
kpt_dir = "../../DanceDataset/dance52/openpose_json/*.json"

kpt_jsons = sorted(glob.glob(kpt_dir))
# save_dir = "../../DanceDataset/dance15/openpose_img"
# save_dir = "../../DanceDataset/001_12/openpose_img"
save_dir = "../../DanceDataset/dance52/openpose_img"

os.makedirs(save_dir, exist_ok=True)

for kpt_json in kpt_jsons:
    ptsList = util.readkeypointsfile_json(kpt_json)
    ptsList = [util.fix_scale_coords(xx, scale, translate) for xx in ptsList]
    kpt_im = util.renderpose25(ptsList[0], 255 * np.ones((1024,1024,3), dtype='uint8'))  # pose
#     kpt_im = util.renderpose25(ptsList[0], 255 * np.ones((512,512,3), dtype='uint8'))  # pose
    kpt_im = util.renderhand(ptsList[2], kpt_im, threshold = 0.05)
    kpt_im = util.renderhand(ptsList[3], kpt_im, threshold = 0.05)  # [h, w, 3]
    kpt_im = cv2.resize(kpt_im, (loadSize, loadSize))
    
    save_path = os.path.join(save_dir, os.path.basename(kpt_json).replace('json', 'jpg'))
    imageio.imwrite(save_path, kpt_im)
print("keypoints render finished")