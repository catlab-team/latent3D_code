# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import tensorflow as tf
import numpy as np
import pickle
import menpo.io as mio
import PIL
import inspect

import torch
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
)
import torchvision.transforms as T

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

# Filter args dictionary with the elements that required by given function
def filter_function_args(args_dict, function):
    function_args = inspect.getfullargspec(function)[0]

    filtered_args_dict = {}
    for key, value in args_dict.items():
        if key in function_args:
            filtered_args_dict[key] = value

    return filtered_args_dict

#----------------------------------------------------------------------------
# Export and Render utils
def from_UV_2_3D(uv):
    info_dict = mio.import_pickle('models/snapshots/512_UV_dict.pkl')        
    tc_ps = info_dict['tcoords_pixel_scaled']
    # trilist = info_dict['trilist']
    #uv = interpolaton_of_uv_xyz(uv,tmask).as_unmasked()
    
    x = uv[0][(tc_ps.points.astype(int).T[0,:], tc_ps.points.astype(int).T[1,:])]
    y = uv[1][(tc_ps.points.astype(int).T[0,:], tc_ps.points.astype(int).T[1,:])] 
    z = uv[2][(tc_ps.points.astype(int).T[0,:], tc_ps.points.astype(int).T[1,:])]
    points = torch.hstack((x.T[:,None],y.T[:,None],z.T[:,None]))

    return points

def get_2D_render_batch(vertices, faces, texture_raw, texture_vertices, normals, image_size, angles, is_save_obj=False, obj_names=None, subdir=None, device=None):
    texture_object = TexturesUV(texture_raw, faces, texture_vertices)
    mesh = Meshes(verts=vertices, faces=faces, textures=texture_object, verts_normals=normals)
    # mesh = Meshes(verts=vertices, faces=faces, textures=texture_object)

    if is_save_obj:
        obj_subdir = os.path.join(subdir, "mesh")
        if not os.path.isdir(obj_subdir):
            os.mkdir(obj_subdir)

        for i in range(int(len(obj_names)/len(angles))):
            obj_path = os.path.join(obj_subdir, f"{obj_names[i]}.obj")
            save_obj(obj_path, vertices[i], faces[i], verts_uvs=texture_vertices[i], faces_uvs=faces[i], texture_map=texture_raw[i])
        

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    R_list = []
    T_list = []
    for angle in angles:
        R, T = look_at_view_transform(2.0, 0, angle) 
        R_list.append(R)
        T_list.append(T)
    
    cameras = FoVPerspectiveCameras(device=device, R=torch.cat(R_list, 0), T=torch.cat(T_list, 0))

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=2, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    # Now move the light so it is on the +Z axis which will be behind the cow. 
    # lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]], ambient_color=((0.58, 0.58, 0.58),), diffuse_color=((0.7, 0.7, 0.7),), specular_color=((0, 0, 0),))
    
    
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
            
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    images = renderer(mesh)
    return images[:, ..., :3]

def export_results(images, subdir, minibatch_size, png_idx=0, save_img=False, image_size=512, img_names=None, angles=[3], is_save_obj=False, device=None):
    UV_dict = mio.import_pickle('models/snapshots/512_UV_dict.pkl')
    texture_vertices = torch.tensor(UV_dict['tcoords'].points).to(device)
    faces = torch.tensor(UV_dict['trilist']).to(device)

    # If there is more than one camera angle
    images = torch.cat([images for _ in range(len(angles))], 0)

    vertices_batch = []
    texture_raw_batch = []
    normals_batch = []
    for i in range(minibatch_size*len(angles)):
        texture_raw = torch.clamp(images[i, 0:3]/2+0.5, 0, 1).permute(1, 2, 0) # 3,512,512 to 512,512,3
        img_shape = T.functional.gaussian_blur(images[i, 3:6], kernel_size=(17, 17), sigma=3)
        vertices = from_UV_2_3D(img_shape)
        normals = images[i, 6:9]
        normals_norm = (normals - normals.min()) / (normals.max() - normals.min())
        normals_norm = from_UV_2_3D(normals_norm)

        texture_raw_batch.append(texture_raw)
        vertices_batch.append(vertices)
        normals_batch.append(normals_norm)

    faces_batch = [faces for _ in range(minibatch_size*len(angles))] # replicate faces 
    texture_vertices_batch = [texture_vertices for _ in range(minibatch_size*len(angles))] # replicate texture_vertices

    vertices_batch = torch.stack(vertices_batch)
    texture_raw_batch = torch.stack(texture_raw_batch)
    faces_batch = torch.stack(faces_batch)
    texture_vertices_batch = torch.stack(texture_vertices_batch)
    normals_batch = torch.stack(normals_batch)

    rendered_images = get_2D_render_batch(vertices_batch, faces_batch, texture_raw_batch, texture_vertices_batch, normals_batch, image_size, angles, is_save_obj=is_save_obj, obj_names=img_names, subdir=subdir, device=device)

    filenames = []
    if save_img:
        for i, rendered_image in enumerate(rendered_images):
            im = PIL.Image.fromarray(np.uint8(rendered_image.cpu().detach().numpy()*255)).convert('RGB')
            if img_names:
                im_path = os.path.join(subdir, f"{img_names[i]}.png")
            else:
                im_path = os.path.join(subdir, '%06d.png' % (png_idx * minibatch_size + i))
            im.save(im_path)
            filenames.append(im_path)

    return rendered_images

def export_model_outputs(images, inter_latent, result_subdir, mode, export_outputs=True, export_inter_latent=True):
    img_subdir = os.path.join(result_subdir, "model_output")
    if not os.path.isdir(img_subdir):
        os.mkdir(img_subdir)
    
    if export_inter_latent:
        with open(f"{img_subdir}/{mode}_inter_latent.npy", 'wb') as file:
            np.save(file, inter_latent)
    
    if export_outputs:
        save_pkl(images[0][0:3], f"{img_subdir}/{mode}_texture.pickle")
        save_pkl(images[0][3:6], f"{img_subdir}/{mode}_normals.pickle")
        save_pkl(images[0][6:9], f"{img_subdir}/{mode}_shape.pickle")



#----------------------------------------------------------------------------
# Tensorflow utils
def init_tf(config_dict=dict()):
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        create_session(config_dict, force_as_default=True)

# Create tf.Session based on config dict of the form
# {'gpu_options.allow_growth': True}
def create_session(config_dict=dict(), force_as_default=False):
    config = tf.ConfigProto()
    for key, value in config_dict.items():
        fields = key.split('.')
        obj = config
        for field in fields[:-1]:
            obj = getattr(obj, field)
        setattr(obj, fields[-1], value)

    session = tf.Session(config=config)
    if force_as_default:
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session

def close_session():
    sess = tf.get_default_session()
    sess.close()
    print('Exiting...')

#----------------------------------------------------------------------------
# Logging of stdout and stderr to a file.

class OutputLogger(object):
    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename, mode='wt'):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()

class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush
 
    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()

output_logger = None

def init_output_logging():
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)

def set_output_log_file(filename, mode='wt'):
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)

#----------------------------------------------------------------------------
# Reporting results.

def create_result_subdir(result_dir, desc, text_list=None, lambda_id=None, lambda_l2=None, num_epochs=None, lr=None, inter_choice=None, random_seed=None, target_image=None):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print("Saving results to", result_subdir)
    set_output_log_file(os.path.join(result_subdir, 'optimize.log'))
    
    print("\nConfiguration:")
    print(f"text: {text_list}\nlambda_id: {lambda_id}\nlambda_l2: {lambda_l2}\nnum_epochs: {num_epochs}\nlearning_rate: {lr}\ninter_choice: {inter_choice}\nrandom_seed: {random_seed}\ntarget_image: {target_image}\n")

    # Export config.
    # try:
    #     with open(os.path.join(result_subdir, 'params.txt'), 'wt') as fout:
    #         fout.write(f"text: {text_list}\nlambda_id: {lambda_id}\nlambda_l2: {lambda_l2}\nnum_epochs: {num_epochs}\nlearning_rate: {lr}\ninter_choice: {inter_choice}\nrandom_seed: {random_seed}\ntarget_image: {target_image}\n")
    # except:
    #     pass

    return result_subdir