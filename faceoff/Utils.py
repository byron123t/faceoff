import imageio
import numpy as np 
import os
from faceoff import Config
import tensorflow.compat.v1 as tf
from faceoff.Crop import *
from faceoff.Models import get_model
from mtcnn import MTCNN
from keras import backend


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



def face_detection(imgfiles):
    imgs = []
    dets = []
    base_faces = []
    detector = MTCNN()

    for img in imgfiles:
        face, det = crop_face(img, detector)
        if face is not None:
            for f, d in zip(face, det):
                img = np.around(img / 255.0, decimals=12)
                base_faces.append(f)
                imgs.append(img)
                dets.append(d)

    faces = np.squeeze(np.array(base_faces))
    if len(imgs) <= 1:
        faces = np.expand_dims(faces, axis=0)

    return faces, imgs, dets

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


class Evaluate:
    def __init__(self,
                 fr_model,
                 batch_size):
        height = fr_model.image_height
        width = fr_model.image_width
        channels = fr_model.num_channels
        shape = (batch_size, channels, height, width)
        self.input_tensor = tf.placeholder(tf.float32, shape)
        self.embedding = fr_model.predict(self.input_tensor)
        self.batch_size = shape[0]


def compute_distance(person1, person2):
    cos_sim = np.dot(person1, person2) / (np.linalg.norm(person1) * np.linalg.norm(person2))
    distance = np.linalg.norm(person1 - person2)
    cos_sim = np.arccos(cos_sim) / math.pi * 60
    avg = (distance + cos_sim) / 2
    return avg


def compute_embeddings(tf_config, batch_size, faces):
    backend.clear_session()
    tf.reset_default_graph()
    with tf.Session(config=tf_config) as sess:
        fr_model = get_model()
        eval = Evaluate(fr_model=fr_model, batch_size=batch_size)
        embeddings = []
        for p, person in enumerate(faces):
            embeddings.append(sess.run(eval.embedding, feed_dict={eval.input_tensor: np.expand_dims(person, axis=0)}))
    return embeddings


def match(buckets, count, p1, p2):
    found = False
    for key, val in buckets.items():
        if p1 in val or p2 in val:
            if p1 not in val:
                buckets[key].append(p1)
                found = True
            if p2 not in val:
                buckets[key].append(p2)
                found = True
            if found:
                break
            else:
                return buckets, count
    if not found:
        count += 1
        buckets[count] = [p1, p2]
    return buckets, count


def face_recognition(faces, threshold, batch_size, tf_config):
    embeddings = compute_embeddings(tf_config, batch_size, faces)
    buckets = {0:[]}
    means = {}
    count = 0

    for p1, person1 in enumerate(embeddings):
        for p2, person2 in enumerate(embeddings):
            if p1 != p2:
                avg = compute_distance(person1, person2)
                if avg <= threshold:
                    buckets, count = match(buckets, count, p1, p2)

    for key, val in buckets.items():
        person_embedding = []
        for i in val:
            person_embedding.append(embeddings[i])
        means[key] = np.mean(person_embedding, axis=0)

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