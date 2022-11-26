# -----------------------------------------------------------------------------------
# Original code from SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Adapted by: Leonardo Manrique, Sergio I. Rincón and Gabriel F. Gonzalez
# -----------------------------------------------------------------------------------
import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
from tqdm import tqdm
import torch
from skimage import io
import requests
import matplotlib.pyplot as plt
from Architectures.SwinIR import SwinIR as net
from Metrics import Metrics_from_SwinIR as util
from Datasets.dataset import Initial_Dataset as DS
from Attacks import attacks 
from Losses import losses as ls
from torch.nn.parallel import DataParallel, DistributedDataParallel
import csv
import time
from ignite.metrics import SSIM, FID, InceptionScore
from ignite.engine.engine import Engine

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

#-------------- Manejo de archivos------------------------------------

# Función de creación de archivos de formato .csv.
def crear_csv(title, rows_list, name):
    '''Función para crear un CSV auxiliar que contiene los resultados de las pruebas'''
    with open(os.path.join(name+'.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        writer.writerows(rows_list)
        f.close()
def agregar_lineas_csv(name,rows_list):
    with open(os.path.join(name + ".csv"), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows_list)
        f.close()


def save_examples(directory_name, examples, names):
    for i in range(examples.shape[0]):
        image = np.transpose(examples[i].cpu().numpy(),(1,2,0))
        io.imsave(os.path.join(directory_name, 'adv_'+names[i] + ".png"), image)
    return None

def main():
    epsilons = [16/255] #epsilons = [10/255, 12/255, 14/255, 16/255]
    step_sizes = [0.05] # [0.001, 0.005, 0.01, 0.05]
    steps = [20]
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--model_path', type=str,
                        default='Models/R_Swin_IR.pth')
    parser.add_argument('--training_patch_size', type=int, default=48) #64 si uso el SwinIR_02
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--name', type=str, default="Se_olvido_el_nombre")
    args = parser.parse_args()

    #gpu_list = ','.join(str(x) for x in [0,1])
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    #device = torch.device('cuda' if gpu_list is not None else 'cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
        
    model = define_model(args)
    model.eval()
    model = model.to(device)
    #model = DistributedDataParallel(model, device_ids=gpu_list)
    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    total_psnr , total_ssim = 0, 0
    imgnames, HR_paths, LR_paths = get_image_pair(args)
    batch_size = 1
    dataset = DS(LR_paths, HR_paths)
    kwargs = {}
    #loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size//len(gpu_list), shuffle=False, **kwargs)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size , shuffle=False, **kwargs)
    #device = torch.device('cuda')
    #device = "cpu"
    rows = []
    titles = ["Epsilon", "Step Size", "Total_steps", "PSNR", "SSIM"]
    count = 1
    for epsilon in tqdm(epsilons):
        for step_size in tqdm(step_sizes): 
            for total_steps in steps:
                id = 0
                for LR, HR, path in loader:
                    id += 1
                    LR = LR
                    HR = HR
                    id_path = path[0].split("/")[3][:-4]
                    adv_examples = attacks.pgd_SwinIR(model, LR, HR, ls.MSE_loss, total_steps, step_size, float("inf"), epsilon, float("inf"),  window_size, border, args.scale)   
                    #save_examples(os.path.join("Ejemplos adversarios"), adv_examples, [id_path])
                    print(id)
                    with torch.no_grad():
                        _, _, h_old, w_old = LR.size()
                        h_pad = (h_old // window_size + 1) * window_size - h_old
                        w_pad = (w_old // window_size + 1) * window_size - w_old
                        adv_examples = torch.cat([adv_examples, torch.flip(adv_examples, [2])], 2)[:, :, :h_old + h_pad, :]
                        adv_examples = torch.cat([adv_examples, torch.flip(adv_examples, [3])], 3)[:, :, :, :w_old + w_pad]
                        output = model(adv_examples)
                        output = output[..., :h_old * args.scale, :w_old * args.scale]
                        if HR is not None:
                            psnr = util.calculate_psnr_torch(output, HR, crop_border=border).data.cpu().numpy()
                            print(psnr)
                            total_psnr += np.sum(psnr)
                            ssim = SSIM(data_range=1.0)
                            ssim.attach(default_evaluator, 'SSIM')
                            state = default_evaluator.run([[output, HR]])
                            print(state.metrics['SSIM'])
                            ssim_metric = state.metrics['SSIM']
                            total_ssim += ssim_metric
                            """
                            fid = FID()
                            fid.attach(default_evaluator, 'fid')
                            state_fid = default_evaluator.run([[output, HR]])
                            print(state.metrics['fid'])
                            fid_metric = state_fid.metrics['fid']
                            total_fid += fid_metric"""
                    """
                    plt.figure()
                    plt.subplot(131)
                    plt.imshow(np.transpose(HR[0].cpu().numpy(),(2,1,0)))
                    plt.subplot(132)
                    plt.imshow(np.transpose(adv_examples[0].cpu().numpy(),(1,2,0)))
                    plt.subplot(133)
                    plt.imshow(np.transpose(output[0].cpu().numpy(),(2,1,0)))
                    plt.show()"""
                    adv_example_aux = np.transpose(adv_examples[0].cpu().numpy(),(1,2,0))
                    output_aux = np.transpose(output[0].cpu().numpy(),(1,2,0))
                    #io.imsave(os.path.join("Ejemplos adversarios","Imagen"+str(id)+"-"+str(epsilon)+".png"), adv_example_aux)
                    #io.imsave(os.path.join("Ejemplos adversarios","Imagen_"+str(id)+".png"), adv_example_aux)
                    #io.imsave(os.path.join("Salida adversaria","Imagen"+str(id)+"-"+str(epsilon)+".png"), output_aux)
                    

                print('Avg psnr: ', total_psnr/len(dataset), 'db')
                print('Avg ssim: ', total_ssim/len(dataset))
                #print('Avg fid: ', total_fid/len(dataset))
                rows.append([epsilon, step_size, total_steps, total_psnr/len(dataset), total_ssim/len(dataset)])
                rows_list = [[epsilon, step_size, total_steps, total_psnr/len(dataset), total_ssim/len(dataset)]]
                if count==1:
                    crear_csv(titles,rows_list,args.name+"_parcial")
                    count += 1
                else:
                    agregar_lineas_csv(args.name+"_parcial", rows_list)
                total_psnr , total_ssim = 0, 0

    crear_csv(titles, rows, args.name)

def define_model(args):
    model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args):
    save_dir = f'results/swinir_{args.task}_x{args.scale}'
    folder = args.folder_gt
    border = args.scale
    window_size = 8
    return folder, save_dir, border, window_size

def get_image_pair(args):
    folder_paths_HR = glob.glob(os.path.join(args.folder_gt, '*.png'))
    ordered_path_list_HR = []
    ordered_path_list_LR = []
    ids_of_images = []
    for i in range(len(folder_paths_HR)):
        path = folder_paths_HR[i]
        (imgname, imgext) = os.path.splitext(os.path.basename(path))
        img_gt = (f'{args.folder_gt}/{imgname}{imgext}')
        img_lq = (f'{args.folder_lq}/{imgname}x{args.scale}{imgext}')
        ordered_path_list_HR.append(img_gt)
        ordered_path_list_LR.append(img_lq)
        ids_of_images.append(ids_of_images)
    return ids_of_images, ordered_path_list_HR, ordered_path_list_LR


if __name__ == '__main__':
    main()

