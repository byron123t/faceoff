import imageio
import numpy as np 
import os
import Config
import tensorflow as tf
from utils.recog import *
from utils.crop import *
from models.face_models import get_model


def set_bounds(params):
    """
    """
    if params['model_type'] == 'small' and params['loss_type'] == 'center':
        pixel_max = 1.0
        pixel_min = -1.0
    else:
        pixel_max = 1.0
        pixel_min = 0.0
    return pixel_max, pixel_min


def transpose_back(params,
                   adv,
                   face):
    """
    """
    if len(adv.shape) == 4:
        adv = adv[0]
    if params['model_type'] == 'small':
        if params['loss_type'] == 'center':
            adv_new_img = (adv + 1.0)/2.0 # scale to [0,1]
            adv_new_img = adv_new_img[::-1,...] # BGR to RGB
            adv_new_img = np.transpose(adv_new_img, (1,2,0)) #CHW to HWC
            face_new = (face + 1.0)/2.0
            face_new = face_new[::-1,...]
            face_new = np.transpose(face_new, (1,2,0))
        elif params['loss_type'] == 'triplet':
            adv_new_img = np.transpose(adv, (1,2,0))
            face_new = np.transpose(face, (1,2,0))
    elif params['model_type'] == 'large':
        adv_new_img = adv
        face_new = face
    return adv_new_img, face_new


def initialize_dict(file_names):
    """
    """
    adv_crop_dict = {}
    delta_clip_dict = {}
    adv_img_dict = {}
    for i in file_names:
        adv_crop_dict[i] = []
        delta_clip_dict[i] = []
        adv_img_dict[i] = []
    return adv_crop_dict, delta_clip_dict, adv_img_dict


def populate_dict(file_names,
                  adv_crop_dict,
                  adv_crop_stack,
                  delta_clip_dict,
                  delta_clip_stack,
                  adv_img_dict,
                  adv_img_stack):
    """
    """
    for i, name in enumerate(file_names):
        adv_crop_dict[name].append(adv_crop_stack[i])
        delta_clip_dict[name].append(delta_clip_stack[i])
        adv_img_dict[name].append(adv_img_stack[i])
    return adv_crop_dict, delta_clip_dict, adv_img_dict


def save_image(file_names,
               out_img_names,
               out_img_names_crop,
               adv_img_stack,
               adv_crop_stack):
    """
    """
    for i, name in enumerate(adv_img_stack):
        if adv_img_stack[i] is not None:
            file = file_names[i]
            imageio.imwrite(out_img_names[file], (adv_img_stack[i] * 255).astype(np.uint8))
            imageio.imwrite(out_img_names_crop[file], (adv_crop_stack[i] * 255).astype(np.uint8))


def save_np(out_npz_names,
            adv_crop_dict,
            delta_clip_dict,
            adv_img_dict):
    """
    """
    for key, val in adv_crop_dict.items():
        if delta_clip_dict[key] is not None:
            np.savez(out_npz_names[key], delta_clip_stack=delta_clip_dict[key])


def load_images(params, dir_names):
    """
    """
    print('Loading Images...')
    faces = {'base': {}, 'source_target': {}}
    file_names = {}
    imgs = {}
    dets = {}
    for name in dir_names:
        print('Loading {}'.format(name))
        imgs[name] = {}
        dets[name] = {}
        file_names[name] = []
        base_faces = []
        
        base_path = os.path.join(Config.ROOT, params['test_dir'], name)
        source_target_path = os.path.join(Config.ROOT, params['align_dir'], name)

        base_files = os.listdir(base_path)
        source_files = os.listdir(source_target_path)

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = create_mtcnn(sess, None)
        for file in base_files:
            img = imageio.imread(os.path.join(base_path, file))
            print(file)
            face, det = crop_face(img, params, pnet, rnet, onet)
            if face is not None:
                img = np.around(img / 255.0, decimals=12)
                file_names[name].append(file)
                base_faces.append(np.array([face]))
                imgs[name][file] = img
                dets[name][file] = det

        temp_files = []
        for file in source_files:
            temp_files.append(os.path.join(source_target_path, file))
        faces['source_target'][name] = read_face_from_aligned(temp_files, params)

        faces['base'][name] = np.squeeze(np.array(base_faces))
        if len(imgs[name]) <= 1:
            faces['base'][name] = np.expand_dims(faces['base'][name], axis=0)

    return faces, file_names, imgs, dets


def load_images_old(params, source, target):
    """
    """
    print('Loading Images...')
    faces = {'base': {}, 'source': {}, 'target': {}}
    file_names = []
    imgs = []
    dets = []
    base_faces = []
        
    base_path = os.path.join(Config.ROOT, params['test_dir'], source)
    source_path = os.path.join(Config.ROOT, params['align_dir'], source)
    target_path = os.path.join(Config.ROOT, params['align_dir'], target)

    base_files = os.listdir(base_path)
    source_files = os.listdir(source_path)
    target_files = os.listdir(target_path)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)
    for file in base_files:
        img = imageio.imread(os.path.join(base_path, file))
        print(file)
        face, det = crop_face(img, params, pnet, rnet, onet)
        if face is not None:
            img = np.around(img / 255.0, decimals=12)
            file_names.append(file)
            base_faces.append(np.array([face]))
            imgs.append(img)
            dets.append(det)

    temp_files = []
    for file in source_files:
        temp_files.append(os.path.join(source_path, file))
    faces['source'] = read_face_from_aligned(temp_files, params)

    temp_files = []
    for file in target_files:
        temp_files.append(os.path.join(target_path, file))
    faces['target'] = read_face_from_aligned(temp_files, params)

    faces['base'] = np.squeeze(np.array(base_faces))
    if len(imgs) <= 1:
        faces['base'] = np.expand_dims(faces['base'], axis=0)

    return faces, file_names, imgs, dets


def return_write_path(params, file_names, target, margin, amplification):
    """
    """
    marg_str = '%0.2f' % margin
    amp_str = '%0.3f' % amplification
    print('Writing images... Margin: {}, Amplification: {}'.format(marg_str, amp_str))
    img_path = {}
    crop_path = {}
    npz_path = {}
    if params['iteration_flag']:
        png_format = '{}_{}_{}_loss_{}_{}_marg_{}_it_{}_amp_{}.png'
        npz_format = '{}_{}_{}_loss_{}_{}_marg_{}_it_{}.npz'
    else:
        png_format = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'
        npz_format = '{}_{}_{}_loss_{}_{}_marg_{}.npz'
    for file in file_names:
        name = file[0 : file.index('.')]
        if params['iteration_flag']:
            png_name = png_format.format(params['attack_name'],
                                         params['model_name'],
                                         params['attack_loss'][0],
                                         name,
                                         target,
                                         marg_str,
                                         params['iterations'],
                                         amp_str)
            npz_name = npz_format.format(params['attack_name'],
                                         params['model_name'],
                                         params['attack_loss'][0],
                                         name,
                                         target,
                                         marg_str,
                                         params['iterations'])
        else:
            png_name = png_format.format(params['attack_name'],
                                         params['model_name'],
                                         params['attack_loss'][0],
                                         name,
                                         target,
                                         marg_str,
                                         amp_str)
            npz_name = npz_format.format(params['attack_name'],
                                         params['model_name'],
                                         params['attack_loss'][0],
                                         name,
                                         target,
                                         marg_str)
        img_path[file] = os.path.join(params['directory_path'], png_name)
        crop_path[file] = os.path.join(params['directory_path_crop'], png_name)
        npz_path[file] = os.path.join(params['directory_path_npz'], npz_name)
    return img_path, crop_path, npz_path
