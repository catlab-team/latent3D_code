import models.tbgan as tbgan
import os
import numpy as np
import tensorflow as tf
import menpo.io as mio
import time

import torch

from config import tbgan_snapshot_name, device, args
from utils import create_result_subdir, close_session, export_model_outputs, export_results, filter_function_args


def generate_random_mesh(grid_size=[1, 1], minibatch_size=8, num_pngs=8, result_dir='./results', desc='random-meshes-32', render=True):
    # Set random seed
    seed = np.random.choice(range(1000))
    random_state = np.random.RandomState(seed)

    # Load model
    tbgan_model = tbgan.load_model(tbgan_snapshot_name)
    #print(Gs.print_layers())

    latents_in = "Gs/latents_in:0"
    labels_in = "Gs/labels_in:0"
    images_out = "Gs/images_out:0"
    inter_layer = "Gs/128x128/Conv1/PixelNorm/mul:0"

    result_subdir = create_result_subdir(result_dir, desc)

    lsfm_tcoords = mio.import_pickle('models/snapshots/512_UV_dict.pkl')['tcoords']
    lsfm_params = []

    for png_idx in range(int(num_pngs/minibatch_size)):
        print('Generating latent vectors...')
        # latents (size: (n, 1536))
        latents = random_state.randn(np.prod(grid_size)*minibatch_size, *tbgan_model.input_shape[1:]).astype(np.float32)  
        labels = np.zeros([latents.shape[0], 7], np.float32)

        sess = tf.get_default_session()
        images, inter_latent = sess.run([images_out, inter_layer], feed_dict={latents_in: latents, labels_in: labels})
        print(np.array(images).shape, np.array(inter_latent).shape)
        # rendered_images = export_results(images, result_subdir, png_idx, minibatch_size)

        # Flatten intermediate latent vectors
        inter_latent = np.array(inter_latent)

    mio.export_pickle(lsfm_params, os.path.join(result_subdir, 'lsfm_params.pkl'))
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

    close_session()

    return latents

def generate_mesh_from_latent_vector(inter_choice, result_subdir, minibatch_size):
    # Load the pre-trained model
    tbgan_model = tbgan.load_model(tbgan_snapshot_name)
    
    model_output_dir = f"{result_subdir}/model_output"

    with open(f"{model_output_dir}/original_inter_latent.npy", 'rb') as file:
        original_inter_latent = np.load(file)
    
    with open(f"{model_output_dir}/final_inter_latent.npy", 'rb') as file:
        final_inter_latent = np.load(file)

    # Choose Layers
    images_name = "Gs/images_out:0"

    if inter_choice == "dense":
        inter_layer_name = "Gs/4x4/Dense/PixelNorm/mul:0"   
    if inter_choice == "conv":
        inter_layer_name = "Gs/4x4/Conv/PixelNorm/mul:0"

    sess = tf.get_default_session()

    original_images_val = sess.run(images_name, feed_dict={
            inter_layer_name: original_inter_latent,
    })
    original_images_torch = torch.tensor(original_images_val, requires_grad=False).to(device)
    export_results(original_images_torch, result_subdir, minibatch_size, angles=[3], img_names=["original"], save_img=False, is_save_obj=True, device=device)
    export_model_outputs(original_images_val, original_inter_latent, result_subdir, "original", export_outputs=True, export_inter_latent=False)


    final_images_val = sess.run(images_name, feed_dict={
            inter_layer_name: final_inter_latent,
    })
    final_images_torch = torch.tensor(final_images_val, requires_grad=False).to(device)
    export_results(final_images_torch, result_subdir, minibatch_size, angles=[3], img_names=["final"], save_img=False, is_save_obj=True, device=device)
    export_model_outputs(final_images_val, final_inter_latent, result_subdir, "final", export_outputs=True, export_inter_latent=False)

    close_session()


if __name__ == "__main__":
    start_time = time.time()

    if args.functionality == "random":
        # Filter args dictionary with the elements that required by given function
        filtered_args_dict = filter_function_args(vars(args), generate_random_mesh)
        generate_random_mesh(**filtered_args_dict)
    
    elif args.functionality == "latent_vector":
        # Filter args dictionary with the elements that required by given function
        filtered_args_dict = filter_function_args(vars(args), generate_mesh_from_latent_vector)
        generate_mesh_from_latent_vector(**filtered_args_dict)

    print(f"Duration: {time.time() - start_time}")