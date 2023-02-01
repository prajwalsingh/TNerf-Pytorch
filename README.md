# DNerf-Pytorch
Implementation of paper D-Nerf: Neural Randiance Fields for Dynamic Scenes

<center>

| Nerf (v18)  |      T-Nerf (v2)    |
|----------|:-------------:|
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/dyn_lego_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/dyn_lego_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/hellwarrior_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/hellwarrior_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/bouncingball_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/bouncingball_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/hook_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/hook_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/jumpingjack_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/jumpingjack_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/mutant_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/mutant_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/trex_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/trex_2.gif"/> |
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/standup_1.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/standup_2.gif"/> |

</center>

## Novel Views
<center>

| Camera  |  Time  | Camera + Time |
|----------|-------------|:-------------:|
| <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/lego_camera.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/lego_time.gif"/> |  <img src="https://github.com/prajwalsingh/DNerf-Pytorch/blob/main/results/lego_time+camera.gif"/> | 

</center>


## Hyperparameters:
<center>

| Parameters  |      Values   |
|----------|:-------------:|
| Iteration | 200K |
| Scheduler | Exponential Decay |
| Scheduler Step | 160K approx. |
| Rays Sample | 1024 |
| Crop | 0.5 |
| Pre Crop Iter | 50 |
| Factor | 2 |
| Near Plane | 2.0 |
| Far Plane | 6.0 |
| Height | 800 / factor |
| Width | 800 / factor |
| Downscale | 2 |
| lr | 5e-4 |
| lrsch_gamma | 0.1 |
| Pos Enc Dim | 10 |
| Dir Enc Dim | 4 |
| Num Samples | 64 |
| Num Samples Fine | 128 |
| Net Dim | 256 |
| Net Depth | 8 |
| Inp Feat | 2*(num_channels*pos_enc_dim) + num_channels |
| Dir Feat | 2*(num_channels*dir_enc_dim) + num_channels |
| Time Feat| 2*(1*dir_enc_dim) + 1 |

</center>

## References:

[1] Computer Graphics and Deep Learning with NeRF using TensorFlow and Keras [![Link](https://pyimagesearch.com/2021/11/17/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-2/)]

[2] 3D volumetric rendering with NeRF [![Link](https://keras.io/examples/vision/nerf/)]

[3] Nerf Official Colab Notebook [![Link](https://colab.research.google.com/drive/1L6QExI2lw5xhJ-MLlIwpbgf7rxW7fcz3#scrollTo=31sNNVves8C2)]

[4] NeRF PyTorch [![Link](https://github.com/sillsill777/NeRF-PyTorch)] ( Special Thanks :) )

[5] PyTorch Image Quality (PIQ) [![Link](https://piq.readthedocs.io/en/latest/index.html)]
