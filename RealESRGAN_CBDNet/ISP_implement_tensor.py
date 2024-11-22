import numpy as np
import cv2
import os
import math
import random
import torch
import torch.nn.functional as F
import scipy.io
import torchvision

def masks_CFA_Bayer(shape, pattern='RGGB'):
    pattern = pattern.upper()
    B, C, H, W = shape
    R_mask = np.zeros((B, C, H, W))
    G_mask = np.zeros((B, C, H, W))
    B_mask = np.zeros((B, C, H, W))

    # Populate masks according to Bayer pattern
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        if channel == 'R':
            R_mask[:, :, y::2, x::2] = 1
        elif channel == 'G':
            G_mask[:, :, y::2, x::2] = 1
        elif channel == 'B':
            B_mask[:, :, y::2, x::2] = 1

    # Convert masks to boolean type
    R_mask = R_mask.astype(bool)
    G_mask = G_mask.astype(bool)
    B_mask = B_mask.astype(bool)

    return R_mask, G_mask, B_mask

def demosaicing_CFA_Bayer_Malvar2004_tensor(CFA, pattern='RGGB'):
    B, C, H, W = CFA.shape
    device = CFA.device
    R_m, G_m, B_m = masks_CFA_Bayer((B, C, H, W), pattern)

    GR_GB = torch.tensor(
        [[0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]],device=device, dtype=torch.float32) / 8

    Rg_RB_Bg_BR = torch.tensor(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, -1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]],device=device, dtype=torch.float32) / 8

    Rg_BR_Bg_RB = Rg_RB_Bg_BR.t()

    Rb_BB_Br_RR = torch.tensor(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]],device=device, dtype=torch.float32) / 8

    
    R_m = torch.tensor(R_m, device=device, dtype=torch.float32)
    G_m = torch.tensor(G_m, device=device, dtype=torch.float32)
    B_m = torch.tensor(B_m, device=device, dtype=torch.float32)

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G = torch.where(torch.logical_or(R_m == 1, B_m == 1), F.conv2d(CFA, GR_GB.unsqueeze(0).unsqueeze(0), padding=2), G)

    RBg_RBBR = F.conv2d(CFA, Rg_RB_Bg_BR.unsqueeze(0).unsqueeze(0), padding=2)
    RBg_BRRB = F.conv2d(CFA, Rg_BR_Bg_RB.unsqueeze(0).unsqueeze(0), padding=2)
    RBgr_BBRR = F.conv2d(CFA, Rb_BB_Br_RR.unsqueeze(0).unsqueeze(0), padding=2)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR
    R_r = (torch.any(R_m == 1, dim=3, keepdim=True) * torch.ones(R.shape,device=device)).float()
    R_c = (torch.any(R_m == 1, dim=2, keepdim=True) * torch.ones(R.shape,device=device)).float()
    B_r = (torch.any(B_m == 1, dim=3, keepdim=True) * torch.ones(B.shape,device=device)).float()
    B_c = (torch.any(B_m == 1, dim=2, keepdim=True) * torch.ones(B.shape,device=device)).float()

    R = torch.where(torch.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = torch.where(torch.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = torch.where(torch.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = torch.where(torch.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = torch.where(torch.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = torch.where(torch.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    RGB = torch.cat((R, G, B), dim=1)
    return RGB

class ISP_tensor:
    def __init__(self, curve_path='./RealESRGAN_CBDNet/'):
        filename = os.path.join(curve_path, '201_CRF_data.mat')
        CRFs = scipy.io.loadmat(filename)
        self.I = torch.tensor(CRFs['I'], dtype=torch.float32)
        self.B = torch.tensor(CRFs['B'], dtype=torch.float32)
        filename = os.path.join(curve_path, 'dorfCurvesInv.mat')
        inverseCRFs = scipy.io.loadmat(filename)
        self.I_inv = torch.tensor(inverseCRFs['invI'], dtype=torch.float32)
        self.B_inv = torch.tensor(inverseCRFs['invB'], dtype=torch.float32)
        self.xyz2cam_all = torch.tensor(
            [[1.0234, -0.2969, -0.2266, -0.5625, 1.6328, -0.0469, -0.0703, 0.2188, 0.6406],
             [0.4913, -0.0541, -0.0202, -0.613, 1.3513, 0.2906, -0.1564, 0.2151, 0.7183],
             [0.838, -0.263, -0.0639, -0.2887, 1.0725, 0.2496, -0.0627, 0.1427, 0.5438],
             [0.6596, -0.2079, -0.0562, -0.4782, 1.3016, 0.1933, -0.097, 0.1581, 0.5181]],
            dtype=torch.float32
        )

    def ICRF_Map(self, input_img, index=0):

        invI_temp = self.I_inv[index, :].clone()  # [bins]
        invB_temp = self.B_inv[index, :].clone()  # [bins]
        invI_temp = invI_temp.to(input_img.device)
        invB_temp = invB_temp.to(input_img.device)


        img = input_img.clone()
        B, C, H, W = img.shape
        img = img.permute(0, 2, 3, 1)  # [B, H, W, C]


        img_flat = img.reshape(-1, C)


        start_bins = torch.floor(img_flat / (9.7656e-04) - 1).clamp(min=1).long()


        for ch in range(C):
            temp_img = img_flat[:, ch] 


            idx = torch.searchsorted(invB_temp, temp_img.contiguous(), right=True)
            idx = idx.clamp(max=invI_temp.size(0) - 1)


            prev_idx = idx - 1
            prev_idx = prev_idx.clamp(min=0)
            

            diff_current = (invB_temp[idx] - temp_img).abs()
            diff_prev = (invB_temp[prev_idx] - temp_img).abs()
            
            idx = torch.where(diff_prev < diff_current, prev_idx, idx)
            

            img_flat[:, ch] = invI_temp[idx]


        img_mapped = img_flat.reshape(B, H, W, C)
        img_mapped = img_mapped.permute(0, 3, 1, 2)  # [B, C, H, W]

        return img_mapped
    
    def CRF_Map(self, input_img, index=0):

        I_temp = self.I[index, :].clone()  # [bins]
        B_temp = self.B[index, :].clone()  # [bins]
        I_temp = I_temp.to(input_img.device)
        B_temp = B_temp.to(input_img.device)


        img = input_img.clone()
        B, C, H, W = img.shape
        img = img.permute(0, 2, 3, 1)  # [B, H, W, C]


        img_flat = img.reshape(-1, C)


        start_bins = torch.floor(img_flat / (9.7656e-04) - 1).clamp(min=1).long()


        for ch in range(C):
            temp_img = img_flat[:, ch]  


            idx = torch.searchsorted(I_temp, temp_img.contiguous(), right=True)
            idx = idx.clamp(max=I_temp.size(0) - 1)


            prev_idx = idx - 1
            prev_idx = prev_idx.clamp(min=0)
            

            diff_current = (I_temp[idx] - temp_img).abs()
            diff_prev = (I_temp[prev_idx] - temp_img).abs()
            
            idx = torch.where(diff_prev < diff_current, prev_idx, idx)
            

            img_flat[:, ch] = B_temp[idx]


        img_mapped = img_flat.reshape(B, H, W, C)
        img_mapped = img_mapped.permute(0, 3, 1, 2)  

        return img_mapped

    def RGB2XYZ(self, input_img):

        img = input_img.clone()
    

        M = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], dtype=torch.float32, device=img.device)
        M = M.permute(1,0)



        def gamma_decode(c):
            c = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            return c
        

        img = gamma_decode(img)


        img = img.permute(0, 2, 3, 1)  # [B, H, W, C]
        

        xyz = torch.einsum('bijc,cd->bijd', img, M)  # [B, H, W, C]
        

        xyz = xyz.permute(0, 3, 1, 2)  # [B, C, H, W]

        return xyz

    def XYZ2RGB(self, input_img):
        img = input_img.clone()
    

        M_inv = torch.tensor([
            [ 3.24048134, -1.53715152, -0.49853633],
            [-0.96925495,  1.87599   ,  0.04155593],
            [ 0.05564664, -0.20404134,  1.05731107]
        ], dtype=torch.float32, device=img.device)
        M_inv = M_inv.permute(1,0)

        img = img.permute(0, 2, 3, 1)  # [B, H, W, C]


        rgb = torch.einsum('bijc,cd->bijd', img, M_inv)  # [B, H, W, C]


        def gamma_encode(c):
            c = torch.where(c <= 0.0031308, c * 12.92, 1.055 * torch.pow(c, 1.0 / 2.4) - 0.055)
            return c

        rgb = gamma_encode(rgb)

        rgb = torch.clamp(rgb, 0.0, 1.0)

        rgb = rgb.permute(0, 3, 1, 2)  # [B, C, H, W]

        return rgb

    def XYZ2CAM(self, input_img, M_xyz2cam=0):
        self.xyz2cam_all = self.xyz2cam_all.to(input_img.device)
        img = input_img.clone()
        img = img.permute(0,2,3,1)
        B, H, W, C = img.shape
        if type(M_xyz2cam) is int:
            cam_index = torch.rand(4, device=img.device)
            cam_index /= cam_index.sum()
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0] + 
                         self.xyz2cam_all[1, :] * cam_index[1] + 
                         self.xyz2cam_all[2, :] * cam_index[2] + 
                         self.xyz2cam_all[3, :] * cam_index[3])
            self.M_xyz2cam = M_xyz2cam
        M_xyz2cam = M_xyz2cam.reshape(3, 3).to(img.device)
        M_xyz2cam /= M_xyz2cam.sum(dim=1, keepdim=True)
        cam = self.apply_cmatrix(img, M_xyz2cam)
        cam = torch.clamp(cam,0,1)
        cam = cam.permute(0,3,1,2)
        return cam

    def CAM2XYZ(self, input_img, M_xyz2cam=0):
        img = input_img.clone()
        img = img.permute(0,2,3,1)
        B, H, W, C = img.shape
        if type(M_xyz2cam) is int:
            cam_index = torch.rand(4, device=img.device)
            cam_index /= cam_index.sum()
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0] +
                         self.xyz2cam_all[1, :] * cam_index[1] +
                         self.xyz2cam_all[2, :] * cam_index[2] +
                         self.xyz2cam_all[3, :] * cam_index[3])
        M_xyz2cam = M_xyz2cam.reshape(3, 3).to(img.device)
        M_xyz2cam /= M_xyz2cam.sum(dim=1, keepdim=True)
        M_cam2xyz = torch.linalg.inv(M_xyz2cam)
        xyz = self.apply_cmatrix(img, M_cam2xyz)
        xyz = torch.clamp(xyz, 0, 1)
        xyz = xyz.permute(0,3,1,2)
        return xyz

    def apply_cmatrix(self, img, matrix):
        B, H, W, C = img.shape
        img_flat = img.reshape(-1, C)
        result_flat = img_flat @ matrix.T
        result = result_flat.reshape(B, H, W, C)
        return result

    def BGR2RGB(self, img):
        img = img[:, [2, 1, 0], :, :]
        return img

    def RGB2BGR(self, img):
        img = img[:, [2, 1, 0], :, :]
        return img

    def mosaic_bayer(self, rgb, pattern='BGGR'):
        B, C, H, W = rgb.shape
        num = torch.zeros(4, dtype=torch.int64)
        temp = self.find(pattern, 'R')
        num[temp] = 0
        temp = self.find(pattern, 'G')
        num[temp] = 1
        temp = self.find(pattern, 'B')
        num[temp] = 2

        mosaic = torch.zeros((B, 1, H, W), device=rgb.device)
        mosaic[:, :, 0::2, 0::2] = rgb[:, num[0], 0::2, 0::2].unsqueeze(1)
        mosaic[:, :, 0::2, 1::2] = rgb[:, num[1], 0::2, 1::2].unsqueeze(1)
        mosaic[:, :, 1::2, 0::2] = rgb[:, num[2], 1::2, 0::2].unsqueeze(1)
        mosaic[:, :, 1::2, 1::2] = rgb[:, num[3], 1::2, 1::2].unsqueeze(1)
        return mosaic

    def WB_Mask(self, img, pattern, fr_now, fb_now):
        wb_mask = torch.ones_like(img)

        if pattern == 'RGGB':
            wb_mask[:, :, 0::2, 0::2] = fr_now
            wb_mask[:, :, 1::2, 1::2] = fb_now
        elif pattern == 'BGGR':
            wb_mask[:, :, 1::2, 1::2] = fr_now
            wb_mask[:, :, 0::2, 0::2] = fb_now
        elif pattern == 'GRBG':
            wb_mask[:, :, 0::2, 1::2] = fr_now
            wb_mask[:, :, 1::2, 0::2] = fb_now
        elif pattern == 'GBRG':
            wb_mask[:, :, 1::2, 0::2] = fr_now
            wb_mask[:, :, 0::2, 1::2] = fb_now

        return wb_mask

    def add_PG_noise(self, img, sigma_s='RAN', sigma_c='RAN'):
        B, C, H, W = img.shape
        min_log = torch.log(torch.tensor([0.0001]))

        if sigma_s == 'RAN':
            sigma_s = min_log + torch.rand(1) * (torch.log(torch.tensor([0.16])) - min_log)
            sigma_s = torch.exp(sigma_s)
        else:
            sigma_s = torch.tensor([sigma_s])
        sigma_s = sigma_s.to(img.device)

        if sigma_c == 'RAN':
            sigma_c = min_log + torch.rand(1) * (torch.log(torch.tensor([0.06])) - min_log)
            sigma_c = torch.exp(sigma_c)
        else:
            sigma_c = torch.tensor([sigma_c])
        sigma_c = sigma_c.to(img.device)

        sigma_total = torch.sqrt(sigma_s * img + sigma_c)
        
        noisy_img = img + sigma_total * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0, 1)

        return noisy_img

    def find(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]
    
    def Demosaic(self, bayer, pattern='BGGR'):
        results = demosaicing_CFA_Bayer_Malvar2004_tensor(bayer, pattern)
        results = torch.clamp(results, 0, 1)
        return results

    def cbdnet_noise_generate_srgb(self, img,sigma_s='RAN',sigma_c='RAN'):
        B, C, H, W = img.shape
        assert C == 3, "Input image must have 3 channels (RGB)."

        img_rgb = img.clone()

        # -------- INVERSE ISP PROCESS -------------------
        # Step 1 : inverse tone mapping
        icrf_index = random.randint(0, 200)
        img_L = self.ICRF_Map(img_rgb, index=icrf_index)
        # Step 2 : from RGB to XYZ
        img_XYZ = self.RGB2XYZ(img_L)
        
        # Step 3: from XYZ to Cam
        img_Cam = self.XYZ2CAM(img_XYZ, M_xyz2cam=0)
        img_Cam = img_Cam
        
        # Step 4: Mosaic
        pattern_index = random.randint(0, 3)
        patterns = ['GRBG', 'RGGB', 'GBRG', 'BGGR']
        pattern = patterns[pattern_index]
        self.pattern = pattern
        
        img_mosaic = self.mosaic_bayer(img_Cam, pattern=pattern)

        # Step 5: inverse White Balance
        min_fc = 0.75
        max_fc = 1
        self.fr_now = random.uniform(min_fc, max_fc)
        self.fb_now = random.uniform(min_fc, max_fc)
        wb_mask = self.WB_Mask(img_mosaic, pattern, self.fr_now, self.fb_now)
        img_mosaic = img_mosaic * wb_mask
        gt_img_mosaic = img_mosaic.clone()

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = self.add_PG_noise(img_mosaic, sigma_s=sigma_s, sigma_c=sigma_c)
        # -------- ISP PROCESS --------------------------
        # Step 5 : White Balance
        wb_mask = self.WB_Mask(img_mosaic_noise, pattern, 1/self.fr_now, 1/self.fb_now)
        img_mosaic_noise = img_mosaic_noise * wb_mask
        img_mosaic_noise = torch.clamp(img_mosaic_noise, 0, 1)
        
        img_mosaic_gt = gt_img_mosaic * wb_mask
        img_mosaic_gt = torch.clamp(img_mosaic_gt, 0, 1)

        # Step 4 : Demosaic
        img_demosaic = self.Demosaic(img_mosaic_noise, pattern=self.pattern)
        img_demosaic_gt = self.Demosaic(img_mosaic_gt, pattern=self.pattern)
        
        # Step 3 : from Cam to XYZ
        img_IXYZ = self.CAM2XYZ(img_demosaic, M_xyz2cam=self.M_xyz2cam)
        img_IXYZ_gt = self.CAM2XYZ(img_demosaic_gt, M_xyz2cam=self.M_xyz2cam)
        
        # Step 2 : from XYZ to RGB
        img_IL = self.XYZ2RGB(img_IXYZ)
        img_IL_gt = self.XYZ2RGB(img_IXYZ_gt)
        
        # Step 1 : tone mapping
        img_Irgb = self.CRF_Map(img_IL, index=icrf_index)
        img_Irgb_gt = self.CRF_Map(img_IL_gt, index=icrf_index)

        
        return img_Irgb_gt, img_Irgb
    
if __name__ == '__main__':
    isp = ISP_tensor()
    path = './imgs/hq00000.png'
    img = cv2.resize(cv2.imread(path),[512,256])
    np.array(img, dtype='uint8')
    img = img.astype('double') / 255.0   
    img = torch.tensor(img,dtype = torch.float32).permute(2,0,1).unsqueeze(0)
    img_rgb = isp.BGR2RGB(img)

    img_Irgb_gt, img_Irgb = isp.cbdnet_noise_generate_srgb(img_rgb)  #gt[B,C,H,W]  #noise[B,C,H,W]
    out = torch.cat([img_rgb,img_Irgb,img_Irgb_gt],dim=3)
    
    torchvision.utils.save_image(out, 'out.png')

    

    
    