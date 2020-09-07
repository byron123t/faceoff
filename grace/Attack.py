import numpy as np
from grace import Config
from grace.Utils import save_image, apply_delta, transpose_back
import argparse


def amplify(params,
            face,
            delta,
            amp,
            person,
            done_imgs):
    """
    Description

    Keyword arguments:
    """
    for i, f in enumerate(face):
        img = person['base']['img'][i]
        filename = person['base']['filename'][i]
        det = person['base']['dets'][i]

        if delta[i] is not None:
            if filename in done_imgs:
                img = done_imgs[filename]

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
                adv_img = apply_delta(delta_clip, img, det, params)  ## BEWARE!!!!!!! of not squaring the amplification
            else:
                adv_img = apply_delta(delta_clip[0], img, det, params)  ## BEWARE!!!!!!! of not squaring the amplification

            done_imgs[filename] = adv_img

        elif filename not in done_imgs:
            done_imgs[filename] = img
        print(done_imgs)

    return done_imgs


# ATTENTION!
# 'adv' and 'delta' are RGB for triplet model, BGR for center face model --> verify brian
def find_adv(sess,
             params,
             fr_model,
             face,
             face_stack_source,
             face_stack_target,
             margin=0):
    from faceoff.CW import CW
    from faceoff.PGD import PGD
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
        print(face.shape)
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
                 people,
                 index):
    import tensorflow as tf
    from keras import backend
    from faceoff.Models import get_model
    """
    Description

    Keyword arguments:
    """
    tf_config = Config.set_gpu('0')
    
    person = people[index]
    if len(person['base']['face']) > 0:
        backend.clear_session()
        tf.reset_default_graph()
        with tf.Session(config=tf_config) as sess:
            fr_model = get_model()

            adv, delta, lp, const = find_adv(sess,
                                             params=params,
                                             fr_model=fr_model,
                                             face=person['base']['face'],
                                             face_stack_source=person['base']['source'],
                                             face_stack_target=person['target'],
                                             margin=params['margin'])
        backend.clear_session()
        tf.reset_default_graph()
        print(person['base']['index'])
    return person, delta


