from backend import Config
import os
import numpy as np
from backend.Attack import outer_attack
from backend.Utils import face_recognition, match_closest
from multiprocessing import Pool


def _face_recognition(faces, threshold, batch_size):
    with Pool(processes=1) as pool:
        return pool.apply(face_recognition, (faces, threshold, batch_size))


def _face_attack(params, person):
    with Pool(processes=1) as pool:
        return pool.apply(outer_attack, (params, person))


def attack_listener(params, person, sess_id):
    done_imgs = {}
    person, delta = _face_attack(params, person)
    return person, delta


def recognize_listener(base_faces, filenames, dets, imgs, counts, img_map, sess_id):
    faces = np.squeeze(np.array(base_faces))
    if len(faces.shape) <= 3:
        faces = np.expand_dims(faces, axis=0)
    embeddings, buckets, means = _face_recognition(faces, 13, 10)
    pairs = match_closest(means)
    lfw_pairs = {}
    for key, val in pairs.items():
        lfw_pairs[key] = {'pair': val, 'faces': buckets[key]}
    filedets = {}
    filedims = {}
    count = 0
    print(len(filenames), len(dets), img_map)
    for f, d, i in zip(filenames, dets, range(len(filenames))):
        if f not in filedets:
            filedets[f] = {}
            filedims[f] = {}
        filedims[f]['height'] = imgs[img_map[i]].shape[0]
        filedims[f]['width'] = imgs[img_map[i]].shape[1]
        filedets[f][count] = d
        count += 1
    np.savez(os.path.join(Config.UPLOAD_FOLDER, sess_id + 'data.npz'), pairs=lfw_pairs, dets=filedets, filenames=filenames, counts=counts)
    return filedets, filedims
