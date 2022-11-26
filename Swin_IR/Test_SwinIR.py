# -----------------------------------------------------------------------------------
# Original code from SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Adapted by: Leonardo Manrique and Sergio I. Rincón
# -----------------------------------------------------------------------------------
import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from Swin_IR.Architectures.SwinIR import SwinIR as net
from Swin_IR.Metrics import Metrics_from_SwinIR as util
import matplotlib.pyplot as plt 

        
def main(proceso,tipo_modelo, directorio, path_LR):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if tipo_modelo == "base":
        model_path = "Swin_IR/Models/SwinIR_01.pth"
    else:
        model_path = "Swin_IR/Models/R_Swin_IR.pth"
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
        
    model = define_model(tipo_modelo)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, border, window_size = setup(directorio)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    psnr, ssim = 0, 0
    if proceso == 'test':
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # read image
            imgname, img_lq, img_gt = get_image_pair(path, directorio)  # image to HWC-BGR, float32
            img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

            # inference
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img_lq, model)
                output = output[..., :h_old * 2, :w_old * 2]

            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

            # evaluate psnr/ssim/psnr_b
            if img_gt is not None:
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = img_gt[:h_old * 2, :w_old * 2, ...]  # crop gt
                img_gt = np.squeeze(img_gt)

                psnr = util.calculate_psnr(output, img_gt, crop_border=border)
                ssim = util.calculate_ssim(output, img_gt, crop_border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(idx, imgname, psnr, ssim))
            else:
                print('Testing {:d} {:20s}'.format(idx, imgname))

        # summarize psnr/ssim
        if img_gt is not None:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            print('\n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(ave_psnr, ave_ssim))
    else:
        path_HR = obtener_HR(path_LR, directorio)
        img_gt = cv2.imread(path_HR, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(path_LR, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model)

            output = output[..., :h_old * 2, :w_old * 2]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_lq = img_lq.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * 2, :w_old * 2, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            print('PSNR: '+str(psnr)+ " dB")
            print('SSIM: '+str(ssim))
            plt.figure()
            plt.subplot(131)
            plt.title("LR (256 x 256)")
            plt.imshow(np.transpose(img_lq,(1,2,0)))
            plt.axis("off")
            plt.subplot(132)
            plt.title("HR Ground Truth  (512 x 512)")
            plt.imshow(np.transpose(img_gt,(0,1,2)))
            plt.axis("off")
            plt.subplot(133)
            plt.imshow(np.transpose(output,(0,1,2)))
            plt.title("HR Predicted (512 x 512)")
            plt.axis("off")
            plt.show()
        

def obtener_HR(path_LR, directorio):
    id_image = os.path.splitext(os.path.basename(path_LR))[0][:-2]
    print(id_image)
    if directorio == "Image Pairs":
        image_gt = os.path.join("Swin_IR","testsets", "Test_Image_Pairs_Swin_IR", "HR", id_image + ".png")
    else:
        image_gt = os.path.join("Swin_IR","testsets", "Test_Flickr_Swin_IR", "HR", id_image + ".png")
    return image_gt


def define_model(tipo_modelo):
    model = net(upscale=2, in_chans=3, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'
    if tipo_modelo == "base":
        pretrained_model = torch.load("Swin_IR/Models/SwinIR_01.pth")
    else:
        pretrained_model = torch.load("Swin_IR/Models/R_Swin_IR.pth")
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(directorio):
    if directorio== "Image Pairs":
        folder = os.path.join("Swin_IR","testsets", "Test_Image_Pairs_Swin_IR", "HR")
    else:
        folder = os.path.join("Swin_IR","testsets", "Test_Flickr_Swin_IR", "HR")

    border = 2
    window_size = 8
    return folder, border, window_size


def get_image_pair(path, directorio):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    if directorio == "Image Pairs":
        img_lq = cv2.imread(f'{os.path.join("Swin_IR","testsets", "Test_Image_Pairs_Swin_IR", "LR")}/{imgname}x{"2"}{imgext}', cv2.IMREAD_COLOR).astype(
            np.float32) / 255.
    else:
        img_lq = cv2.imread(f'{os.path.join("Swin_IR","testsets", "Test_Flickr_Swin_IR", "LR")}/{imgname}x{"2"}{imgext}', cv2.IMREAD_COLOR).astype(
            np.float32) / 255.

    return imgname, img_lq, img_gt

def test(img_lq, model):
    output = model(img_lq)
    return output

"""
if __name__ == '__main__':
    main("e","robusto", "Image Pairs", os.path.join("testsets", "Test_Image_Pairs_Swin_IR", "LR", "20170221_131655x2.png"))"""


