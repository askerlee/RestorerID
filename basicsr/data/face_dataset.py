import random
import time
import os
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import cv2
import numpy as np
import scipy
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from transformers import CLIPProcessor
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.utils import face_align

@DATASET_REGISTRY.register()
class FFHQAndVGGFaceDataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQAndVGGFaceDataset, self).__init__()
        self.opt = opt

        self.ffhq_folder = opt['ffhqdataroot_gt']
        self.vggface_folder = opt['vggfacedataroot_gt']

        self.ffhqpaths = [osp.join(self.ffhq_folder, f'{v:05d}.png') for v in range(70000)]
        self.vggfacepaths = self.prepare_vggfacepaths()

        self.paths = self.ffhqpaths + self.vggfacepaths
    
    def prepare_vggfacepaths(self):
        image_files_list = []
        train_txt_path = osp.join(self.vggface_folder, 'train.txt')
        with open(train_txt_path, 'r') as file:
            train_folders = file.read().splitlines()
        for folder_name in train_folders:
            folder_path = osp.join(self.vggface_folder, "VGGface2_HQ",folder_name)
            for image_file in os.listdir(folder_path):
                image_files_list.append(osp.join(folder_path,image_file))
        return image_files_list

    def __getitem__(self, index):

        gt_path = self.paths[index]
        img_gt = cv2.imread(gt_path)
        img_gt = img_gt.astype(np.float32) / 255.
        img_gt = cv2.resize(img_gt,(512,512))

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        

        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



@DATASET_REGISTRY.register()
class CelebAndVGGFaceDataset(data.Dataset):
    def __init__(self, opt):
        super(CelebAndVGGFaceDataset, self).__init__()
        self.opt = opt

        self.Celeb_dir = opt['Celebdataroot_gt']
        self.vggface_dir = opt['vggfacedataroot_gt']

        self.Celeb_img_folders = self.prepare_Celebfolders()
        self.vggface_img_folders = self.prepare_vggfacefodlers()

        self.img_folders = self.Celeb_img_folders + self.vggface_img_folders
        self.totensor_transform = transforms.ToTensor()

        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(512,512))
    
    
    def prepare_vggfacefodlers(self):
        image_folders_list = []
        train_txt_path = osp.join(self.vggface_dir, 'train.txt')
        with open(train_txt_path, 'r') as file:
            train_folders = file.read().splitlines()
        for folder_name in train_folders:
            img_folder_path = osp.join(self.vggface_dir, "VGGface2_HQ_Clean",folder_name)
            image_folders_list.append(img_folder_path)
        return image_folders_list
    
    def prepare_Celebfolders(self):
        image_folders_list = []
        train_txt_path = osp.join(self.Celeb_dir, 'train.txt')
        with open(train_txt_path, 'r') as file:
            train_folders = file.read().splitlines()
        for folder_name in train_folders:
            folder_path = osp.join(self.Celeb_dir, "Celeb_Ref_Clean",folder_name)
            image_folders_list.append(folder_path)
        return image_folders_list
    
    def __getitem__(self,index):
        imgfolder = self.img_folders[index]
        filenames = os.listdir(imgfolder)
        ref_num = min(2,len(filenames))
        selected_files = random.sample(filenames, ref_num)
        
        gt_path = osp.join(imgfolder,selected_files[0])
        img_gt = cv2.imread(gt_path)
        img_gt = img_gt.astype(np.float32) / 255.
        img_gt = cv2.resize(img_gt,(512,512))
        img_gt = augment(img_gt, hflip=False, rotation=False)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)

        ref_imgpath = osp.join(imgfolder,selected_files[1])
        ref_img = cv2.imread(ref_imgpath)
        faces = self.app.get(ref_img)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding)
        ref_image = face_align.norm_crop(ref_img, landmark=faces[0].kps, image_size=224)
        


        return {'img_gt': img_gt, 'img_ref':ref_image, 'faceid_embeds':faceid_embeds}

        
    def __len__(self):
        return len(self.img_folders)


