# TS - Diffusion Distillation

This repository provides an extension of the Diffusion TS model, proposed in https://github.com/Y-debug-sys/Diffusion-TS. 

Distillation of the originally trained model leads to significantly faster Sampling, as sampling steps are reduced in half at each iteration of Distillation. The training procedure follows from what was proposed in "Progressive Distillation for Fast Sampling of Diffusion Models", by Tim Salimans and Jonathan Ho https://openreview.net/forum?id=TIdIXIpzhoI . 


## Acknowledgements

I made use of the following github repos for their valuable code base:

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/cientgu/VQ-Diffusion

https://github.com/XiangLi1999/Diffusion-LM

https://github.com/philipperemy/n-beats

https://github.com/salesforce/ETSformer

https://github.com/ermongroup/CSDI

https://github.com/jsyoon0823/TimeGAN
