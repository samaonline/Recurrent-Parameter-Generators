# Overview
This the code for paper "Recurrent Parameter Generator".

# ImageNet with ResNet-RPG
Training (you can choose architecure from superresnet18e and superresnet34e)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py -j 32 -a superresnet18e /dataset/imagenet/ -b 512 --savename res18e_lr0.3
```
Evaluate (make sure --savename matches)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py -j 32 -a superresnet18e /dataset/imagenet/ -b 512 --savename res18e_lr0.3 -e --resume res18e_lr0.3.pth.tarbest.pth.tar
```
