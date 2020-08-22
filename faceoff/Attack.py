import numpy as np
import tensorflow as tf
from faceoff import Config
from faceoff.CW import CW
from faceoff.PGD import PGD
from faceoff.Models import get_model
from faceoff.Crop import apply_delta
from faceoff.Utils import transpose_back
from faceoff.Utils import save_image
import argparse
from keras import backend


def amplify(params,
            face,
            delta,
            amp,
            dets,
            imgs,
            file_names):
    """
    Description

    Keyword arguments:
    """
    adv_crop_stack = []
    adv_img_stack = []
    delta_clip_stack = []
    prev = None
    prev_adv = None
    for i, f in enumerate(face):
        if delta[i] is not None:
            if imgs[i] == prev:
                use_img = prev_adv
            
            cur_delta = delta[i] * amp
            cur_face = face[i]
            
            adv_crop = cur_face + cur_delta
            adv_crop = np.maximum(adv_crop, params['pixel_min'])
            adv_crop = np.minimum(adv_crop, params['pixel_max'])

            adv_crop, temp_face = transpose_back(params=params,
                                                 adv=adv_crop,
                                                 face=cur_face)

            delta_clip = adv_crop - temp_face
            if len(delta_clip.shape) == 3:
                adv_img = apply_delta(delta_clip, 1, use_img, dets[i], params)  ## BEWARE!!!!!!! of not squaring the amplification
            else:
                adv_img = apply_delta(delta_clip[0], 1, use_img, dets[i], params)  ## BEWARE!!!!!!! of not squaring the amplification
            if len(delta_clip.shape) != 3:
                adv_crop_stack.append(adv_crop[0])
                delta_clip_stack.append(delta_clip[0])
            else:
                adv_crop_stack.append(adv_crop)
                delta_clip_stack.append(delta_clip)
            adv_img_stack.append(adv_img)
            prev = imgs[i]
            prev_adv = adv_img
        else:
            adv_crop_stack.append(None)
            adv_img_stack.append(None)
            delta_clip_stack.append(None)
    return adv_crop_stack, delta_clip_stack, adv_img_stack


# ATTENTION!
# 'adv' and 'delta' are RGB for triplet model, BGR for center face model --> verify brian
def find_adv(sess,
             params,
             fr_model,
             face,
             face_stack_source,
             face_stack_target,
             margin=0):
    """
    Description

    Keyword arguments:
    """
    num_base = face.shape[0]
    num_src = face_stack_source.shape[0]
    num_target = face_stack_target.shape[0]

    if params['attack'] == 'CW':
        cw_attack = CW(sess=sess,
                       model=fr_model,
                       params=params,
                       num_base=num_base,
                       num_src=num_src,
                       num_target=num_target,
                       confidence=margin,
                       margin=margin)
        best_lp, best_const, best_adv, best_delta = cw_attack.attack(face,
                                                                     target_imgs=face_stack_target,
                                                                     src_imgs=face_stack_source,
                                                                     params=params)
    elif params['attack'] == 'PGD':
        #pending verification
        best_lp = []
        best_const = []
        best_adv = []
        best_delta = []
        if params['batch_size'] <= 0:
            batch_size = num_base
        else:
            batch_size = min(params['batch_size'], num_base)
        for i in range(0,len(face),batch_size):
            pgd_attack = PGD(fr_model, back='tf', sess=sess)
            pgd_params = {'eps': params['epsilon'], 
                          'eps_iter': params['epsilon_steps'], 
                          'nb_iter': params['iterations'], 
                          'ord': params['norm']}
            pgd_attack.set_parameters(params=params,
                                      target_imgs=face_stack_target,
                                      src_imgs=face_stack_source,
                                      margin=margin, 
                                      model=fr_model,
                                      base_imgs=face[i:i+batch_size],
                                      **pgd_params)

            adv, lp = pgd_attack.generate(face[i:i+batch_size], **pgd_params)
            
            # best_adv = sess.run(best_adv)
            delta = adv - face[i:i+batch_size]
            const = [None] * face.shape[0]

            best_lp.extend(best_lp)
            best_const.extend(const)
            best_adv.extend(adv)
            best_delta.extend(delta)


    return best_adv, best_delta, best_lp, best_const


def outer_attack(params,
                 faces,
                 file_names,
                 source,
                 target,
                 tf_config,
                 imgs,
                 dets):
    """
    Description

    Keyword arguments:
    """
    for margin in params['margin_list']:
        backend.clear_session()
        tf.reset_default_graph()
        with tf.Session(config=tf_config) as sess:
            Config.BM.mark('Model Loaded')
            fr_model = get_model(params)
            Config.BM.mark('Model Loaded')

            Config.BM.mark('Adversarial Example Generation')
            adv, delta, lp, const = find_adv(sess, 
                                             params=params, 
                                             fr_model=fr_model,
                                             face=faces['base'], 
                                             face_stack_source=faces['source'],
                                             face_stack_target=faces['target'],
                                             margin=margin)
        Config.BM.mark('Adversarial Example Generation')

        Config.BM.mark('Dictionary Initialization')
        adv_crop_dict, delta_clip_dict, adv_img_dict = initialize_dict(file_names=file_names)
        Config.BM.mark('Dictionary Initialization')

        Config.BM.mark('Amplifying and Writing Images')
        for amplification in params['amp_list']:
            img_path, crop_path, npz_path = return_write_path(params=params,
                                                              file_names=file_names,
                                                              target=target,
                                                              margin=margin,
                                                              amplification=amplification)
            adv_crop_stack, delta_clip_stack, adv_img_stack = amplify(params=params,
                                                                      face=faces['base'],
                                                                      delta=delta,
                                                                      amp=amplification,
                                                                      dets=dets,
                                                                      imgs=imgs,
                                                                      file_names=file_names)
            save_image(file_names = file_names,
                       out_img_names = img_path,
                       out_img_names_crop = crop_path,
                       adv_img_stack = adv_img_stack,
                       adv_crop_stack = adv_crop_stack)

        Config.BM.mark('Amplifying and Writing Images')
