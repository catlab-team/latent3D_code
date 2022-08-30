import os
import numpy as np
import torch
import argparse
from utils import init_output_logging, EasyDict, init_tf

#----------------------------------------------------------------------------
# Argument parse
parser = argparse.ArgumentParser()

# Common arguments
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument('--minibatch_size', type=int, default=1)
parser.add_argument('--result_dir', type=str, default="./results")
parser.add_argument('--folder_title', type=str, default="")

# optimize.py
parser.add_argument("--mode", type=str, choices=['text-based', 'image-based'], default="text-based")

parser.add_argument('--text_list', nargs='+')
parser.add_argument('--image_path', type=str)

parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument('--lambda_id', type=float, default=0.01)
parser.add_argument('--lambda_l2', type=float, default=0.)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--inter_choice', type=str, choices=['dense', 'conv'], default="dense")

# pca.py
###

# sample.py 
parser.add_argument('--functionality', type=str, choices=["random", "latent_vector"], default="random")
parser.add_argument('--result_subdir', type=str, default="./results/000-optimize")


args = parser.parse_args()

#----------------------------------------------------------------------------
# Seed initilizations
np.random.seed(args.seed)

#----------------------------------------------------------------------------
# Logging initilizations
init_output_logging()

#----------------------------------------------------------------------------
# TensorFlow initilizations
tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = False      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
# tf_config['gpu_options.visible_device_list']          = '1'     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

os.environ.update(env)
init_tf(config_dict=tf_config)

#----------------------------------------------------------------------------
# TBGAN initilizations
tbgan_snapshot_name = "network-snapshot-013600.pkl"

#----------------------------------------------------------------------------
# Torch initilizations
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

#----------------------------------------------------------------------------
# CLIP templates
clip_templates = [
    'a bad photo of a {}.',
    # 'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    # 'a tattoo of a {}.',
    # 'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    # 'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    # 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    # 'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    # 'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    # 'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    # 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    # 'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    # 'a tattoo of the {}.',    
    'a 3d object of the {}.',
    'a 3d object of a {}.',
    'a 3d face of a {}.',
    'a 3d face of the {}.',
    "a face of a {}.",
    "a face of the {}."
]