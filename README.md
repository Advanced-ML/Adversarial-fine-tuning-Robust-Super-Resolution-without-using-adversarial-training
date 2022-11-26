# Adversarial fine tuning: Robust Super-Resolution without using adversarial training

Gabriel Gonzalez, Leonardo Manrique and Sergio Rincón


Este repositorio contiene las implementaciones asociadas al documento 'Adversarial fine tuning: Robust Super-Resolution without using adversarial training'. En ese sentido, se cuenta con 3 arquitecturas: SwinIR, SRGCNN y Latent Diffusion (adaptado). Es importante mencionar que mientras que SwinIR y Latent Difussion están implementadas en python 3, SRGCNN está implementada sobre python 2 y por consiguiente tiene requerimientos diferentes. A continuación se detallan los requerimientos.

### SwinIR
Los requerimientos de SwinIR se detallan en éste [repositorio](https://github.com/cszn/KAIR) dentro de un archivo .txt 

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
La ruta de los datasets y de los pesos de los modelos se encuentra a continuaciónw

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
