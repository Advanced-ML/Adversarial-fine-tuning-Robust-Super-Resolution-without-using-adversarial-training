# Adversarial fine tuning: Robust Super-Resolution without using adversarial training

Gabriel Gonzalez, Leonardo Manrique and Sergio Rincón


This repository contains the implementations associated with the document 'Adversarial fine tuning: Robust Super-Resolution without using adversarial training'. In that sense, there are 3 architectures: SwinIR, SRGCNN and Latent Diffusion (adapted). It is important to mention that while SwinIR and Latent Diffusion are implemented on Python 3, SRGCNN is implemented on Python 2 and therefore has different requirements. 

Before start create an empty directory with the name testsets, an empty directory with the name logs, an empty directory with the name Models and an empty directory with the name checkpoint. 

### SwinIR
The SwinIR requirements are detailed in this [repository](https://github.com/cszn/KAIR) within a .txt file. Create an empty directory with the name testsets within the SwinIR directory.


### Latent Diffusion
Run the following commands associated with the installation of packages. Be carefull, the taming transformer folder must be inside of /LDM. Create an empty directory with the name logs within the directory LDM.
```
!git clone https://github.com/CompVis/taming-transformers
!pip install -e ./taming-transformers
!pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
```
### ESRGCNN
ESRGCNN requirements are detailed in the [official repository](https://github.com/hellloxiaotian/ESRGCNN). It is recommended to install them
in a new different enviroment. 

## Dataset and weights
Latent diffusion weights are automatically downloaded when used. The SRGCNN (esrgcnn.pth) model is located in /media/disk0/sirincon/Final Project Files/Models and must be placed inside the Models folder. Regarding to SwinIR, create a folder with the name Models within the SwinIR folder and put in that location the following SwinIR models.  The SwinIR (SwinIR_01.pth) and Robust SwinIR (R_Swin_IR.pth) models are located in /media/disk0/sirincon/Final Project Files/Models and must be placed inside the /Swin_IR/Models folder.

Regarding the datasets, they are located in /media/disk0/sirincon/Final Project Files/Datasets, where there are six folders. There are two folders with images named specifically for every architecture. The folders Test_Image_Pairs_CNN and Test_Flickr_CNN must be placed inside the /testsets folder. The Test_Image_Pairs_Swin_IR and Test_Flickr_Swin_IR folders must be placed inside the /Swin_IR/testsets folder. The Image_Pairs_75_256_512 and Flickr_75_256_512 must be placed inside the /LDM folder.

## Testing

Once the requirements have been installed, the following commands can be used to reproduce the results reported in the main document.

To reproduce the results in SwinIR and Robust SwinIR with the two datasets:
```
python main.py --mode test
```

To reproduce the results in SwinIR and Robust SwinIR with only one image from Image Pairs dataset (Notice that the image_name does not have .png extention):
```
python main.py --mode demo --img image_name --dir image_directory (Flickr or Image Pairs)
```

To reproduce the results in Latent Diffusion with the two datasets:
```
python main.py --mode test --model LD
```

To reproduce the results in Latent Diffusion with only one image from Image Pairs dataset:
```
python main.py --mode demo --model LD --img image_name
```

To reproduce the results in SRGCNN (requires python 2) with the two datasets:
```
python main.py --mode test --model CNN
```

To reproduce the results in SRGCNN (requires python 2) with only one image from Image Pairs dataset:
```
python main.py --mode demo --model CNN --img image_name --dir image_directory (Flickr or Image Pairs)
```
If you want to test any image from Flickr2K dataset, add this argument to the command:
```
python --dir Flickr
```

