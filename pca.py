import models.tbgan as tbgan
import os
import time
import numpy as np
import tensorflow as tf
import menpo.io as mio
from sklearn.decomposition import PCA

from config import tbgan_snapshot_name, clip_templates, args
from utils import load_pkl, save_pkl, create_result_subdir, close_session


def create_pca_components(run_id, snapshot=None, grid_size=[1, 1], minibatch_size=8, num_samples=1000, n_components=10):
    # Set random seed
    seed = np.random.choice(range(1000))
    random_state = np.random.RandomState(seed)

    # Load model
    tbgan_model = tbgan.load_model(tbgan_snapshot_name)

    # Choose layers
    latents_in = "Gs/latents_in:0"
    labels_in = "Gs/labels_in:0"
    inter_layer = "Gs/128x128/Conv1/PixelNorm/mul:0"

    # Create random z and exp
    all_latents = random_state.randn(np.prod(grid_size)*num_samples, *tbgan_model.input_shape[1:]).astype(np.float32)  
    all_labels = np.zeros([all_latents.shape[0], 7], np.float32)

    all_inter_latent = []

    for png_idx in range(int(num_samples/minibatch_size)):
        start = time.time()
        print('Generating samples %d-%d / %d... in ' % (png_idx*minibatch_size, (png_idx+1)*minibatch_size, num_samples), end='')
        latents = all_latents[png_idx*minibatch_size : (png_idx+1)*minibatch_size]
        labels = all_labels[png_idx*minibatch_size : (png_idx+1)*minibatch_size]

        sess = tf.get_default_session()
        inter_latent = sess.run(inter_layer, feed_dict={latents_in: latents, labels_in: labels})
        print(inter_latent.shape)
        print(len(inter_latent))
        all_inter_latent.extend(inter_latent) # concatenate two lists  
        print('%0.2f seconds' % (time.time() - start))

    # PCA
    start = time.time()
    print('Calculating PCA... in ', end='')    
    
    all_inter_latent = np.array(all_inter_latent)
    flat_inter_latent = all_inter_latent.reshape(all_inter_latent.shape[0], -1) # this line depends on layer we choose. if we choose layer with shape (n, 128, 128), we need to flatten it to (n, 128*128) in order to apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(flat_inter_latent)
    pca_comps = pca.components_

    print('%0.2f seconds' % (time.time() - start))

    save_pkl(pca_comps, f"./results/32/pca_comps_{num_samples}_{n_components}")

    close_session()

    return pca_comps

def apply_pca_to_random_meshes(run_id, snapshot=None, grid_size=[1, 1], minibatch_size=8, num_pngs=8, result_dir='./results', desc='apply_pca_to_random_meshes'):
    # Set random seed
    seed = np.random.choice(range(1000))
    random_state = np.random.RandomState(seed)

    # Load model
    tbgan_model = tbgan.load_model(tbgan_snapshot_name)

    print(tbgan_model.print_layers())

    latents_in = "Gs/latents_in:0"
    labels_in = "Gs/labels_in:0"
    images_out = "Gs/images_out:0"
    inter_layer = "Gs/128x128/Conv1/PixelNorm/mul:0"

    result_subdir = create_result_subdir(result_dir, desc)
    lsfm_tcoords = mio.import_pickle('models/snapshots/512_UV_dict.pkl')['tcoords']
    lsfm_params = []

    pca_comps = load_pkl("./results/32/pca_comps_300_10")
    alphas = [300, 500, 1000]  # alpha: manipulation strength

    original_result_subdir = f"{result_subdir}/original_results"
    os.makedirs(original_result_subdir)

    for comp_num in range(len(pca_comps)):
        for alpha in alphas:
            PCA_result_subdir = f"{result_subdir}/PCA_results/comp_{str(comp_num)}/alpha_{str(alpha)}"
            os.makedirs(PCA_result_subdir)


    for png_idx in range(int(num_pngs/minibatch_size)):
        print('Generating latent vectors...')
        latents = random_state.randn(np.prod(grid_size)*minibatch_size, *tbgan_model.input_shape[1:]).astype(np.float32)
        labels = np.zeros([latents.shape[0], 7], np.float32)

        sess = tf.get_default_session()
        # writer = tf.summary.FileWriter('./graphs', graph=sess.graph)

        images, inter_latent = sess.run([images_out, inter_layer], feed_dict={latents_in: latents, labels_in: labels})

        print(np.array(images).shape, np.array(inter_latent).shape)
        
        # export_results(images, original_result_subdir, png_idx, minibatch_size)

        # Flatten intermediate latent vectors
        inter_latent = np.array(inter_latent)
        flat_inter_latent = inter_latent.reshape(inter_latent.shape[0], -1) # this line depends on layer we choose. if we choose layer with shape (n, 128, 128), we need to flatten it to (n, 128*128) in order to apply PCA]

        # Manipulated images via PCA
        for comp_num in range(len(pca_comps)):
            print(f"PCA Components: {comp_num}/{len(pca_comps)}")
            for alpha in alphas:
                PCA_result_subdir = f"{result_subdir}/PCA_results/comp_{str(comp_num)}/alpha_{str(alpha)}"

                flat_inter_latent_manipulated = flat_inter_latent + alpha*pca_comps[comp_num, :]
                inter_latent_manipulated = flat_inter_latent_manipulated.reshape(inter_latent.shape) # invert to original shape

                sess = tf.get_default_session()
                # !!!! the part in feed_dict (latents_in: latents, labels_in: labels) is not necessary, we can set all to zero. (this part needs to be feeded because the graph which used in training needs it but don't use it in testing)
                images = sess.run(images_out, feed_dict={inter_layer: inter_latent_manipulated, latents_in: latents, labels_in: labels})

                # export_results(images, PCA_result_subdir, png_idx, minibatch_size)

    mio.export_pickle(lsfm_params, os.path.join(result_subdir, 'lsfm_params.pkl'))
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

    close_session()

    return latents