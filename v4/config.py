import os
import math

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

dataset_type    = 'real' # 'synthetic', 'real', 'llff'

if dataset_type == 'synthetic':
	# train_camera_path = 'dataset/nerf_synthetic/lego/transforms_train.json' 
	# val_camera_path   = 'dataset/nerf_synthetic/lego/transforms_val.json' 
	# train_image_path  = 'dataset/nerf_synthetic/lego'
	# val_image_path    = 'dataset/nerf_synthetic/lego'
	train_camera_path = 'dataset/dsynthetic/standup/transforms_train.json' 
	val_camera_path   = 'dataset/dsynthetic/standup/transforms_test.json' 
	train_image_path  = 'dataset/dsynthetic/standup'
	val_image_path    = 'dataset/dsynthetic/standup'

	factor       = 2 # factor
	pre_height   = 800
	pre_width    = 800
	image_height = int(pre_height/factor)
	image_width  = int(pre_width/factor)
	downscale    = pre_width / image_width
	near_plane   = 2.0
	far_plane    = 6.0
	epochs       = 1341#4001#3501
	lrsch_step   = 900#3200#2500
	pre_epoch    = 50
	pre_crop     = 0.5
	noise_value  = 0.0

elif (dataset_type == 'real') or (dataset_type == 'llff'):
	# train_camera_path = 'dataset/nerf_real_360/pinecone/poses_bounds.npy'
	# val_camera_path   = 'dataset/nerf_real_360/pinecone/poses_bounds.npy'
	# train_image_path  = 'dataset/nerf_real_360/pinecone/images'
	# val_image_path    = 'dataset/nerf_real_360/pinecone/images'
	# basedir           = 'dataset/nerf_real_360/pinecone/'
	# spherify          = True

	# train_camera_path = 'dataset/nerf_llff_data/trex/poses_bounds.npy'
	# val_camera_path   = 'dataset/nerf_llff_data/trex/poses_bounds.npy'
	# train_image_path  = 'dataset/nerf_llff_data/trex/images'
	# val_image_path    = 'dataset/nerf_llff_data/trex/images'
	# basedir           = 'dataset/nerf_llff_data/trex/'
	# spherify          = False
	
	# train_camera_path = 'dataset/self_360/fishtank/poses_bounds.npy'
	# val_camera_path   = 'dataset/self_360/fishtank/poses_bounds.npy'
	# train_image_path  = 'dataset/self_360/fishtank/images'
	# val_image_path    = 'dataset/self_360/fishtank/images'
	# basedir           = 'dataset/self_360/fishtank/'
	# spherify          = True

	train_camera_path = 'dataset/nvidia_data_full/Balloon1-2/dense/poses_bounds.npy'
	val_camera_path   = 'dataset/nvidia_data_full/Balloon1-2/dense/poses_bounds.npy'
	train_image_path  = 'dataset/nvidia_data_full/Balloon1-2/dense/images'
	val_image_path    = 'dataset/nvidia_data_full/Balloon1-2/dense/images'
	basedir           = 'dataset/nvidia_data_full/Balloon1-2/dense/'
	spherify          = False

	factor       = 4 # factor
	pre_height   = 1024#1080
	pre_width    = 1920#1920
	image_height = int(pre_height/factor)
	image_width  = int(pre_width/factor)
	downscale    = pre_width / image_width
	epochs       = 8341#3501
	lrsch_step   = 7800
	pre_epoch    = 0
	pre_crop     = 0.5
	noise_value  = 0.0


device       = 'cuda'
use_ndc      = False
lr           = 5e-4
num_channels = 3
batch_size   = 1
vis_freq     = 10
ckpt_freq    = 10
lrsch_gamma  = 0.1

pos_enc_dim  = 10
dir_enc_dim  = 4

num_samples  = 64
num_samples_fine  = 128
net_dim      = 256
in_feat      = 2*(num_channels*pos_enc_dim) + num_channels
dir_feat     = 2*(num_channels*dir_enc_dim) + num_channels
time_feat    = 2*(1*dir_enc_dim) + 1
skip_layer   = 4
net_depth    = 8
num_samples_fine  = 2 * num_samples
n_samples    = 1024#4096