# Adversarial fine tuning: Robust Super-Resolution without using adversarial training

Gabriel Gonzalez, Leonardo Manrique and Sergio Rincón


Este repositorio contiene las implementaciones asociadas al documento 'Adversarial fine tuning: Robust Super-Resolution without using adversarial training'. En ese sentido, se cuenta con 3 arquitecturas: SwinIR, SRGCNN y Latent Diffusion (adaptado). Es importante mencionar que mientras que SwinIR y Latent Difussion están implementadas en python 3, SRGCNN está implementada sobre python 2 y por consiguiente tiene requerimientos diferentes. A continuación se detallan los requerimientos asociados a cada arquitectura. 

### SwinIR


### Latent Diffusion


### ESRGCNN




## Dataset and weights
Para facilitar 

## Testing

Una vez instalados los requerimientos, se pueden emplear los siguientes comandos para reproducir los resultados reportados en el documento principal 

Para reproducir los resultados en SwinIR y Robust SwinIR con los test datasets
```
main.py --mode test
```

To reproduce the results in SwinIR and Robust SwinIR con sólo una imagen

```
main.py --mode demo --img image_name.png
```

