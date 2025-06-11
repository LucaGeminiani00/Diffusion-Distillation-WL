# Wavelet Diffusion and Progressive Distillation for Time Series synthesis 

## Project Description 

This repository is the main source code of my master of science thesis at Bocconi University, titled: "Diffusion Wavelet: Interpretable diffusion and Progressive Distillation for time series synthesis". It provides an extension of the Diffusion-TS model, proposed in https://github.com/Y-debug-sys/Diffusion-TS, by changing the underlying logic for learning the Seasonal and Trend components of the time series, exploiting the Wavelet Transform rather than the Fourier Transform. 
It also provides an implementation of Progressive Distillation, extending it to the Time Series domain.
The repository is to be refined for code readability, but it is fully functional. The colab notebook provided can be quickly exploited to train the diffusion models on GPUs. Running the implemented Diffusion models on CPU is not reccomended, especially on datasets with more than 5 features.  

## Progressive Distillation 
Distillation of the originally trained model teaches a student to sample with half the Diffusion steps of the teacher, rendering the original network into a narrower, faster one. The training procedure follows the one originally proposed in "Progressive Distillation for Fast Sampling of Diffusion Models", by Tim Salimans and Jonathan Ho https://openreview.net/forum?id=TIdIXIpzhoI , with slight adaptions due to the use of stochastic samplers. 

## Acknowledgements

For the design of the Diffusion architecture, and of the Wavelet block, I have made use of the following github repos:

https://github.com/fbcotter/pytorch_wavelets

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/Y-debug-sys/Diffusion-TS

https://github.com/cientgu/VQ-Diffusion

https://github.com/XiangLi1999/Diffusion-LM

https://github.com/philipperemy/n-beats

https://github.com/salesforce/ETSformer

https://github.com/ermongroup/CSDI

https://github.com/jsyoon0823/TimeGAN
