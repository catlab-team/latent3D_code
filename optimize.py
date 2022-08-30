from datetime import datetime
import time
import models.tbgan as tbgan
import clip
import numpy as np
import tensorflow as tf
import PIL
from models.arcface import IDLoss

import torch
import torchvision.transforms as T

from config import device, tbgan_snapshot_name, clip_templates, args
from utils import export_model_outputs, export_results, create_result_subdir, close_session, filter_function_args


def clip_loss(clip_model, rendered_images, texts=None, target_image=None):
    preprocess = T.Compose([
         T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC),
         T.CenterCrop(size=(224, 224)),
        #  T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    images = []

    images.append(rendered_images[0].permute(2,0,1))
    images.append(rendered_images[1].permute(2,0,1))
    images.append(rendered_images[2].permute(2,0,1))

    image_input = torch.stack(images).cuda(device)
    image_features = clip_model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    if texts != None:
        text_tokens = clip.tokenize(texts).cuda(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        loss = 1 - torch.mean(image_features @ text_features.t())

    elif target_image != None:
        target_image_torch = torch.stack([target_image]).cuda(device)
        target_image_features = clip_model.encode_image(preprocess(target_image_torch))
        target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)

        loss = 1 - torch.mean(image_features @ target_image_features.t())

    #loss = -1*F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))
    # loss = torch.sum(torch.abs(image_features - text_features))
    return loss


def optimize_latent_vector(seed, mode, minibatch_size, num_epochs, result_dir, lambda_id, lambda_l2, learning_rate, inter_choice, folder_title, text_list=None, image_path=None,):   
    # Load the pre-trained model
    tbgan_model = tbgan.load_model(tbgan_snapshot_name)
    clip_model, _ = clip.load("ViT-B/32", device=device)

    if mode == "text-based":
        texts = []
        for text in text_list:
            texts = [*texts, *[template.format(text) for template in clip_templates]] #format with class

    folder_name = f"o-{mode[0]}-{seed}-{folder_title}-id_{lambda_id}-l2_{lambda_l2}-e_{num_epochs}-lr_{learning_rate}"

    result_subdir = create_result_subdir(result_dir, folder_name, text_list, lambda_id, lambda_l2, num_epochs, learning_rate, inter_choice, seed, image_path)

    # # Choose Layers
    latents_name = "Gs/latents_in:0"
    labels_name = "Gs/labels_in:0"
    images_name = "Gs/images_out:0"

    if inter_choice == "dense":
        inter_layer_name = "Gs/4x4/Dense/PixelNorm/mul:0"   
    if inter_choice == "conv":
        inter_layer_name = "Gs/4x4/Conv/PixelNorm/mul:0"

    inter_layer_tensor = tf.get_default_graph().get_tensor_by_name(inter_layer_name)
    images_tensor = tf.get_default_graph().get_tensor_by_name(images_name)

    # Initialize inputs
    seed = np.random.choice(range(1000))
    random_state = np.random.RandomState(seed)

    init_latent = random_state.randn(minibatch_size, tbgan_model.input_shape[1]).astype(np.float32)  
    # init_latent = random_state.rand(1, *Gs.input_shape[1:]).astype(np.float32)*2 - 1
    init_label = np.zeros(shape=(minibatch_size, 7)).astype(np.float32)

    grad_renderer_ph = tf.placeholder(tf.float32, name="grad_renderer_ph") # gradient came from Pytorch differentiable renderer
    grad = tf.gradients(images_tensor, inter_layer_tensor, grad_ys=grad_renderer_ph)

    sess = tf.get_default_session()

    init_inter_latent = sess.run(inter_layer_name, feed_dict={
        latents_name:init_latent, 
        labels_name: init_label, 
        })   
    with tf.variable_scope('inputs'):
        inter_latent = tf.get_variable("inter_latent", initializer=init_inter_latent, trainable=True)

    lr_ph = tf.placeholder(tf.float32, name='lr_ph')
    with tf.variable_scope('adam'):
        adam = tf.train.AdamOptimizer(lr_ph).apply_gradients(zip(grad, [inter_latent]))

    sess.run(tf.variables_initializer(tf.global_variables('inputs')))
    sess.run(tf.variables_initializer(tf.global_variables('adam')))

    if image_path != None:
        target_image = torch.tensor(np.array(PIL.Image.open(image_path).convert('RGB'))/255., requires_grad=False).to(device)

    id_loss = IDLoss(device)
    for iteration in range(num_epochs):
        images_val = sess.run(images_name, feed_dict={
            inter_layer_name: inter_latent.eval(),
            })
        images_torch = torch.tensor(images_val, requires_grad=True).to(device)

        rendered_images = export_results(images_torch, result_subdir, minibatch_size, angles=[3, 30, -30], save_img=True, image_size=224, img_names=["current_image", "current_left", "current_right"], device=device)
        if iteration==0:
            original_image = rendered_images[0].detach().clone()
            export_model_outputs(images_val, inter_latent.eval(), result_subdir, "original", export_outputs=False)
            export_results(images_torch, result_subdir, minibatch_size, save_img=True, angles=[13, -13, 3, 30, -30], image_size=1024, img_names=["original_13", "original_-13", "original_image", "original_left", "original_right"], is_save_obj=False, device=device)
            

        if mode == "text-based":
            closs = clip_loss(clip_model, [rendered_images[0], rendered_images[1], rendered_images[2]], texts=texts) # give rendering from three different angle 
        elif mode == "image-based":
            closs = clip_loss(clip_model, [rendered_images[0], rendered_images[1], rendered_images[2]], target_image=target_image.permute(2, 0, 1)) # give rendering from three different angle 
        

        iloss = id_loss(original_image.permute(2, 0, 1), rendered_images[0].permute(2, 0, 1))
        l2loss = torch.sum((original_image - rendered_images[0])**2) / (original_image.shape[0]*original_image.shape[1])

        images_torch.retain_grad() # calculate grad of images_torch (needs to say explicity to avoid deleting grad to optimize results)
        loss = closs + lambda_id*iloss + lambda_l2*l2loss
        loss.backward()

        grad_renderer = images_torch.grad.cpu().detach().numpy()

        print(f"[{datetime.now()}] - Iteration {iteration} | Loss: {loss.cpu().detach().numpy():.7f} | Identity Loss: {iloss:.7f} | CLIP Loss {closs:.7f} | L2 Loss {l2loss:.7f}")

        grad_val, _ = sess.run([grad, adam], feed_dict={
            grad_renderer_ph: grad_renderer, 
            lr_ph: learning_rate,
            inter_layer_name: inter_latent.eval(),
            })

    images_val = torch.Tensor(sess.run(images_name, feed_dict={
        inter_layer_name: inter_latent.eval(),
        })).to(device)

    export_model_outputs(images_val, inter_latent.eval(), result_subdir, "final", export_outputs=False)
    rendered_images = export_results(images_val, result_subdir, minibatch_size, angles=[13,-13,3, 30, -30], save_img=True, image_size=1024, img_names=["final_13", "final_-13", "final_image", "final_left", "final_right"], is_save_obj=False, device=device)
    
    close_session()


if __name__ == "__main__":
    start_time = time.time()

    # Filter args dictionary with the elements that required by given function
    filtered_args_dict = filter_function_args(vars(args), optimize_latent_vector)

    optimize_latent_vector(**filtered_args_dict)

    print(f"Duration: {time.time() - start_time}")