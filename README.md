# Adversarial fine tuning: Robust Super-Resolution without using adversarial training

Gabriel Gonzalez, Leonardo Manrique and Sergio Rincón


This repository contains the implementations associated with the document 'Adversarial fine tuning: Robust Super-Resolution without using adversarial training'. In that sense, there are 3 architectures: SwinIR, SRGCNN and Latent Diffusion (adapted). It is important to mention that while SwinIR and Latent Diffusion are implemented on Python 3, SRGCNN is implemented on Python 2 and therefore has different requirements. 


### SwinIR
The SwinIR requirements are detailed in this [repository](https://github.com/cszn/KAIR) within a .txt file


### Latent Diffusion
Correr los siguientes comandos asociados a la instalación de paquetes
```
!git clone https://github.com/CompVis/taming-transformers
!pip install -e ./taming-transformers
!pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
```
### SRGCNN
Los requerimientos de SRGCNN se detallan en el [repositorio oficial](https://github.com/hellloxiaotian/ESRGCNN). Es recomendable instalarlos en un 
enviroment aparte. 

## Dataset and weights
La ruta de los datasets y de los pesos de los modelos se encuentra a continuación

## Testing

Una vez instalados los requerimientos, se pueden emplear los siguientes comandos para reproducir los resultados reportados en el documento principal 

Para reproducir los resultados en SwinIR y Robust SwinIR con los dos datasets:
```
main.py --mode test
```

To reproduce the results in SwinIR and Robust SwinIR con sólo una imagen:
```
main.py --mode demo --img image_name.png
```

Para reproducir los resultados en Latent Diffusion con los dos datasets:
```
main.py --mode test --model LD
```

Para reproducir los resultados en Latent Diffusion con sólo una imagen:
```
main.py --mode demo --model LD --img image_name.png
```

Para reproducir los resultados en SRGCNN (requiere python 2) con los dos datasets:
```
main.py --mode test --model CNN
```

Para reproducir los resultados en SRGCNN (requiere python 2) con sólo una imagen:
```
main.py --mode demo --model CNN --img image_name.png
```
