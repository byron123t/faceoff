import imageio
import numpy as np 
import os
import math
from backend import Config
from backend.Crop import *
from backend.Models import get_model
from tensorflow.keras import backend
from zipfile import ZipFile
import io
from PIL import Image
from functools import partial
from multiprocessing import Pool


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


def save_image(done_imgs,
               sess_id):
    """
    """
    with ZipFile(os.path.join(Config.UPLOAD_FOLDER, '{}.zip'.format(sess_id)), 'w') as zf:
        for filename, img in done_imgs.items():
            buf = io.BytesIO()
            orig = filename.replace(sess_id, '')
            index = orig.index('.')
            im = Image.fromarray((img * 255).astype(np.uint8))
            ext = orig[index:].lower()
            print(ext)
            if ext == '.jpg' or ext == '.jpeg':
                format_type = 'JPEG'
            elif ext == '.png':
                format_type = 'PNG'
            elif ext == '.gif':
                format_type = 'GIF'
            else:
                format_type = 'ERROR'
            im.save(buf, format_type)
            zf.writestr(orig, buf.getvalue())
            print(orig)


def face_detection(imgfiles, outfilenames, sess_id):
    import tensorflow as tf
    from mtcnn import MTCNN
    filenames = []
    imgs = []
    dets = []
    base_faces = []
    detector = MTCNN()
    face, det, count = crop_face(imgfiles, detector, outfilenames, sess_id)
    if face is not None:
        for f, d in zip(face, det):
            img = np.around(imgfiles / 255.0, decimals=12)
            base_faces.append(f)
            imgs.append(img)
            dets.append(d)
            filenames.append(outfilenames)

    print(len(base_faces), filenames, len(imgs), dets, count)
    return base_faces, filenames, img, dets, count


def load_images(params, selected, sess_id):
    """
    """
    print('Loading Images...')
    people = []
    data = np.load(os.path.join(Config.UPLOAD_FOLDER, sess_id + 'data.npz'), allow_pickle=True)
    dets = data['dets']
    pairs = data['pairs']
    filenames = data['filenames']
    counts = data['counts']
    filename_dict = {}
    for file in filenames:
        d = dets.item()
        for i, val in d[file].items():
            filename_dict[i] = file
    print(filename_dict)
    for key, face_matches in pairs.item().items():
        print(face_matches)
        person = {'base': {}}
        person['base']['index'] = []
        person['base']['filename'] = []
        person['base']['img'] = []
        person['base']['face'] = []
        person['base']['source'] = []
        person['base']['dets'] = []
        for i in face_matches['faces']:
            file = filename_dict[i]
            index = file.index('.')
            face = imageio.imread(os.path.join(Config.UPLOAD_FOLDER, '{}{}_{}.png'.format(sess_id, file[:index], counts[i])))
            face = np.around(np.transpose(face, (2,0,1))/255.0, decimals=12)
            face = (face-0.5)*2
            img = imageio.imread(os.path.join(Config.UPLOAD_FOLDER, file))
            img = np.around(img / 255.0, decimals=12)

            if i in selected:
                print(selected, i)
                person['base']['face'].append(face)
                person['base']['index'].append(i)
                person['base']['filename'].append(file)
                person['base']['img'].append(img)
                person['base']['source'].append(face)
                person['base']['dets'].append(dets.item()[file][i])

        if len(person['base']['face']) > 0:
            print(face_matches)
            print(len(person['base']['face']))
            target_path = os.path.join(Config.ROOT, params['align_dir'], face_matches['pair'])
            target_files = os.listdir(target_path)
            temp_files = []
            for file in target_files:
                temp_files.append(os.path.join(target_path, file))
            person['target'] = read_face_from_aligned(temp_files[:1], params)

            person['base']['face'] = np.squeeze(np.array(person['base']['face']))
            if len(person['base']['face'].shape) <= 3:
                person['base']['face'] = np.expand_dims(person['base']['face'], axis=0)
            print(person['base']['face'].shape)
            person['base']['source'] = np.squeeze(np.array(person['base']['source']))
            if len(person['base']['source'].shape) <= 3:
                person['base']['source'] = np.expand_dims(person['base']['source'], axis=0)
            else:
                person['base']['source'] = person['base']['source'][0]
                person['base']['source'] = np.expand_dims(person['base']['source'], axis=0)
            if len(person['target'].shape) <= 3:
                person['target'] = np.expand_dims(person['base']['face'], axis=0)
            people.append(person)

    return people


class Evaluate:
    def __init__(self,
                 fr_model,
                 batch_size):
        import tensorflow as tf
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
    import tensorflow as tf
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
    backend.clear_session()
    tf.reset_default_graph()
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
    new_bucket = {}
    means = {}
    done = []
    count = 0
    for person, neighbors in buckets.items():
        if person not in done:
            people[count] = [embeddings[person]]
            new_bucket[count] = [person]
            for f in neighbors:
                people[count].append(f)
                new_bucket[count].append(f)
            means[count] = np.mean(people[count], axis=0)
            done.extend(neighbors)
            done.append(person)
            count += 1
    return new_bucket, means


def bfs(buckets, size, embeddings):
    visited = [False] * size
    count = 0
    queue = []
    people = {}
    means = {}
    for k, v in buckets.items():
        if not visited[k]:
            queue.append(k)
            visited[k] = True
            people[count] = []
            while queue:
                cur = queue.pop(0)
                people[count].append(cur)
                for i in buckets[cur]:
                    if not visited[i]:
                        queue.append(i)
                        visited[i] = True
            count += 1
    for person, face in people.items():
        embed = []
        for i in face:
            embed.append(embeddings[i])
        means[person] = np.mean(embed, axis=0)
    return people, means


def face_recognition(faces, threshold, batch_size):
    tf_config = Config.set_gpu('0')
    embeddings = compute_embeddings(tf_config, batch_size, faces)
    buckets = {}
    means = {}
    done = []
    for p1, person1 in enumerate(embeddings):
        for p2, person2 in enumerate(embeddings):
            if p1 != p2 and p2 not in done:
                avg = compute_distance(person1, person2)
                if avg <= threshold:
                    print(avg, threshold, p1, p2)
                    buckets = add_bucket(buckets, p1, p2)
                    buckets = add_bucket(buckets, p2, p1)
                    print(buckets)
                else:
                    if p1 not in buckets:
                        buckets[p1] = []
                    if p2 not in buckets:
                        buckets[p2] = []
        done.append(p1)
    if len(done) == 1:
        buckets[0] = []
    print(buckets)
    people, means = bfs(buckets, len(embeddings), embeddings)
    print(people)
    return embeddings, people, means


def match_closest(embeddings):
    npzfile = np.load(os.path.join(Config.ROOT, 'embeddings/lfw_embeddings.npz'), allow_pickle=True)
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
