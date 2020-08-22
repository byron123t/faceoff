import imageio
import numpy as np 
import os
import math
from faceoff import Config
import tensorflow as tf
from faceoff.Crop import *
from faceoff.Models import get_model
from tensorflow.keras import backend


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



def face_detection(imgfiles, outfilenames):
    filenames = []
    imgs = []
    dets = []
    base_faces = []
    count = 0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)

    for i, img in enumerate(imgfiles):
        face, det, count = crop_face(img, pnet, rnet, onet, outfilenames[i], count)
        if face is not None:
            for f, d in zip(face, det):
                img = np.around(img / 255.0, decimals=12)
                base_faces.append(f)
                imgs.append(img)
                dets.append(d)
                filenames.append(outfilenames[i])

    faces = np.squeeze(np.array(base_faces))
    if len(imgs) <= 1:
        faces = np.expand_dims(faces, axis=0)

    return faces, filenames, imgs, dets

def load_images(params, selected, sess_id):
    """
    """
    print('Loading Images...')
    people = []
    data = np.load(os.path.join(Config.UPLOAD_FOLDER, sess_id + 'data.npz'))
    dets = data['dets']
    pairs = data['pairs']
    filenames = data['filenames']

    filename_dict = {}
    for file in filenames:
        for i, val in dets[file].items():
            filename_dict[i] = file

    for lfw, face_matches in pairs.items():
        target_path = os.path.join(Config.ROOT, params['align_dir'], lfw)
        person = {'base': {}}
        person['base']['index'] = []
        person['base']['filename'] = []
        person['base']['img'] = []
        person['base']['face'] = []
        person['base']['dets'] = []

        for i in face_matches:
            file = filename_dict[i]
            index = file.index('.')
            face = imageio.imread(os.path.join(Config.UPLOAD_FOLDER, '{}_{}.png'.format(file[:index], i)))
            face = np.around(np.transpose(face, (2,0,1))/255.0, decimals=12)
            face = (face-0.5)*2
            img = imageio.imread(os.path.join(Config.UPLOAD_FOLDER, file))

            person['base']['index'].append(i)
            person['base']['filename'].append(file)
            person['base']['img'].append(img)
            person['base']['face'].append(face)
            person['base']['dets'].append(dets[file][i])

        temp_files = []
        for file in target_files:
            temp_files.append(os.path.join(target_path, file))
        person['target'] = read_face_from_aligned(temp_files, params)

        person['base']['face'] = np.squeeze(np.array(person['base']['face']))
        if len(imgs) <= 1:
            person['base']['face'] = np.expand_dims(person['base']['face'], axis=0)
        people.append(person)

    return people


class Evaluate:
    def __init__(self,
                 fr_model,
                 batch_size):
        height = fr_model.image_height
        width = fr_model.image_width
        channels = fr_model.num_channels
        shape = (batch_size, channels, width, height)
        self.input_tensor = tf.placeholder(tf.float32, shape)
        self.embedding = fr_model.predict(self.input_tensor)
        self.batch_size = shape[0]


def compute_distance(person1, person2):
    # cos_sim = np.dot(person1, person2) / (np.linalg.norm(person1) * np.linalg.norm(person2))
    distance = np.linalg.norm(person1 - person2)
    # cos_sim = np.arccos(cos_sim) / math.pi * 60
    # avg = (distance + cos_sim) / 2
    return distance


def compute_embeddings(tf_config, batch_size, faces):
    backend.clear_session()
    tf.reset_default_graph()
    with tf.Session(config=tf_config) as sess:
        fr_model = get_model()
        eval = Evaluate(fr_model=fr_model, batch_size=batch_size)
        cur_embeddings = []
        sub_batch = -len(faces)
        for i in range(0, len(faces), batch_size):
            cur_batch = len(faces) - i
            cur_imgs = faces[i:i+eval.batch_size]
            if eval.batch_size > cur_batch:
                sub_batch = eval.batch_size - cur_batch
                cur_imgs = np.pad(cur_imgs, ((0,sub_batch),(0,0),(0,0),(0,0)))
            cur_embeddings.extend(sess.run(eval.embedding, feed_dict={eval.input_tensor: cur_imgs}))
        embeddings = np.array(cur_embeddings[:-sub_batch])
    return embeddings


def add_bucket(buckets, p1, p2):
    if p1 in buckets:
        if p2 not in buckets[p1]:
            buckets[p1].append(p2)
    else:
        buckets[p1] = [p2]
    return buckets


def match_buckets(buckets, embeddings):
    people = {}
    means = {}
    done = []
    count = 0
    for person, neighbors in buckets.items():
        if person not in done:
            people[count] = [embeddings[person]]
            for f in neighbors:
                people[count].append(embeddings[f])
            means[count] = np.mean(people[count], axis=0)
            done.extend(neighbors)
            done.append(person)
            count += 1
    return people, means


def face_recognition(faces, threshold, batch_size, tf_config):
    embeddings = compute_embeddings(tf_config, batch_size, faces)
    buckets = {}
    means = {}
    done = {}
    for p1, person1 in enumerate(embeddings):
        for p2, person2 in enumerate(embeddings):
            if p1 != p2:
                if p1 not in done or p2 not in done[p1] or p1 not in done[p2]:
                    if p1 not in done:
                        done[p1] = [p2]
                    else:
                        done[p1].append(p2)
                    if p2 not in done:
                        done[p2] = [p1]
                    else:
                        done[p2].append(p1)

                    avg = compute_distance(person1, person2)
                    print(avg, threshold)
                    if avg <= threshold:
                        buckets = add_bucket(buckets, p1, p2)
                        buckets = add_bucket(buckets, p2, p1)
                    else:
                        buckets[p1] = []
                        buckets[p2] = []
    people, means = match_buckets(buckets, embeddings)

    return embeddings, buckets, means


def match_closest(embeddings):
    npzfile = np.load(os.path.join(Config.ROOT, 'embeddings/lfw_embeddings.npz'))
    people = npzfile['people']
    means = npzfile['mean']
    pairs = {}

    for person1, mean1 in embeddings.items():
        min_distance = 1e10
        min_person = person1

        for person2, mean2 in zip(people, means):
            avg = compute_distance(mean1, mean2)
            if avg < min_distance:
                min_distance = avg
                min_person = person2

        pairs[person1] = min_person
        print(person1, min_person, min_distance)

    return pairs