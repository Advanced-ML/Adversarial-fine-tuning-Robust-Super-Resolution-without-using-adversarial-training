import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--model",type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "test" and args.model== None:
        import Swin_IR.Test_SwinIR as SwinIR
        print("Test with the original model for Image Pairs dataset")
        SwinIR.main("test", "base", "Image Pairs", 1)
        print("Test with the original model for the Flickr dataset")
        SwinIR.main("test", "base", "Flickr", 1)
        print("Robust model test for Image Pairs dataset")
        SwinIR.main("test", "robust", "Image Pairs", 1)
        print("Test with the robust model for Flickr dataset")
        SwinIR.main("test", "robust", "Flickr", 1)
    elif args.mode == "demo" and args.dir != None and args.img !=None and args.model == None: 
        import Swin_IR.Test_SwinIR as SwinIR
        print("Test with the original Swin IR model")
        if args.dir == "Image Pairs":
            path = os.path.join("Swin_IR", "testsets", "Test_Image_Pairs_Swin_IR", "LR", args.img + "x2.png")
            SwinIR.main("demo", "base", "Image Pairs", path)
        else:
            path = os.path.join("Swin_IR", "testsets", "Test_Flickr_Swin_IR", "LR", args.img + "x2.png")
            SwinIR.main("demo", "base", "Flickr", path)

    elif args.mode == "test" and args.model == "LD":
        import LDM.main as LDM
        print("Test con Latent Diffusion para el conjunto de datos Flickr")
        LDM.final_run("LDM/Flickr_75_256_512", False, 1)
        print("Test con Latent Diffusion para el conjunto de datos de Image Pairs")
        LDM.final_run("LDM/Image_Pairs_75_256_512", False, 1)
    elif args.mode == "demo" and args.model == "LD" and args.img != None:
        import LDM.main as LDM
        print("Test con Latent Diffusion para la imagen")
        LDM.final_run("LDM/Flickr_75_256_512", True, args.img)
    elif args.mode == "test" and args.model== "CNN": 
        import tcw_sample_b as CNN
        print("Test con ESRGCNN en el dataset Flickr")
        CNN.main("test", "Flickr", 1)
        print("Test con ESRGCNN en el dataset Image Pairs")
        CNN.main("test", "Image Pairs", 1)
    elif args.mode == "demo" and args.model== "CNN" and args.img != None and args.dir != None:
        import tcw_sample_b as CNN
        if args.dir == "Image Pairs":
            path = os.path.join("testsets", "Test_Image_Pairs_CNN", "x2", args.img + "_SRF_2_LR.png")
            CNN.main("demo", "Image Pairs", path)
        else:
            path = os.path.join("testsets", "Test_Flickr_CNN", "x2", args.img + "_SRF_2_LR.png")
            CNN.main("demo", "Test_Flickr_CNN", path)
        
if __name__ == "__main__":
    main()
