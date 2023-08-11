# Recurrent Parameter Generator
[[Project]](https://github.com/samaonline/Recurrent-Parameter-Generators) [[Paper]](https://arxiv.org/pdf/2107.07110.pdf) [[Blog]](https://dailyai.github.io/2021-07-16/2107-07110)

## Overview
`Recurrent Parameter Generator (RPG)` is the author's re-implementation of the method described in:  
"[Compact and Optimal Deep Learning with Recurrent Parameter Generators](https://arxiv.org/abs/2107.07110)"   
[Jiayun Wang](http://pwang.pw/),&nbsp; [Yubei Chen](https://redwood.berkeley.edu/people/yubei-chen/),&nbsp;
[Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/),&nbsp; [Brian Cheung](https://redwood.berkeley.edu/people/brian-cheung/),&nbsp; [Yann LeCunn](http://yann.lecun.com/); (UC Berkeley & Meta & NYU & MIT)&nbsp;
in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023

## ImageNet with ResNet-RPG
Training (you can choose architecure from superresnet18e and superresnet34e)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py -j 32 -a superresnet18e /dataset/imagenet/ -b 512 --savename res18e_lr0.3
```
Evaluate (make sure --savename matches)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py -j 32 -a superresnet18e /dataset/imagenet/ -b 512 --savename res18e_lr0.3 -e --resume res18e_lr0.3.pth.tarbest.pth.tar
```

## License and Citation
The use of this software is released under [BSD-3](./LICENSE).
```
@inproceedings{wang2023compact,
  title={Compact and Optimal Deep Learning with Recurrent Parameter Generators},
  author={Wang, Jiayun and Chen, Yubei and Yu, Stella X and Cheung, Brian and LeCun, Yann},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3900--3910},
  year={2023}
}
```
