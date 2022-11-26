# Adversarial fine tuning: Robust Super-Resolution without using adversarial training

Gabriel Gonzalez, Leonardo Manrique and Sergio Rincón


This repository contains the implementations associated with the document 'Adversarial fine tuning: Robust Super-Resolution without using adversarial training'. In that sense, there are 3 architectures: SwinIR, SRGCNN and Latent Diffusion (adapted). It is important to mention that while SwinIR and Latent Diffusion are implemented on Python 3, SRGCNN is implemented on Python 2 and therefore has different requirements. 


### SwinIR
The SwinIR requirements are detailed in this [repository](https://github.com/cszn/KAIR) within a .txt file


### Latent Diffusion
Run the following commands associated with the installation of packages
```
!git clone https://github.com/CompVis/taming-transformers
!pip install -e ./taming-transformers
!pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
```
### SRGCNN
SRGCNN requirements are detailed in the [official repository](https://github.com/hellloxiaotian/ESRGCNN). It is recommended to install them
in a new different enviroment. 

## Dataset and weights
La ruta de los datasets y de los pesos de los modelos se encuentra a continuación

## Testing

Once the requirements have been installed, the following commands can be used to reproduce the results reported in the main document.

To reproduce the results in SwinIR and Robust SwinIR with the two datasets:
```
main.py --mode test
```

To reproduce the results in SwinIR and Robust SwinIR with only one image:
```
main.py --mode demo --img image_name.png
```

To reproduce the results in Latent Diffusion with the two datasets:
```
main.py --mode test --model LD
```

To reproduce the results in Latent Diffusion with only one image:
```
main.py --mode demo --model LD --img image_name.png
```

To reproduce the results in SRGCNN (requires python 2) with the two datasets:
```
main.py --mode test --model CNN
```

To reproduce the results in SRGCNN (requires python 2) with only one image:
```
main.py --mode demo --model CNN --img image_name.png
```
