import numpy as np 
import os
import math
from grace import Config
from zipfile import ZipFile
import io
from PIL import Image
from functools import partial
from multiprocessing import Pool
from mtcnn import MTCNN
import sys
import imageio
import cv2


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


def image_format(filename):
    index = filename.index('.')
    ext = filename[index:].lower()
    if ext == '.jpg' or ext == '.jpeg':
        format_type = 'JPEG'
    elif ext == '.png':
        format_type = 'PNG'
    elif ext == '.gif':
        format_type = 'GIF'
    else:
        format_type = 'ERROR'
    return format_type


def zip_image(filename, sess_id, img):
    with ZipFile(os.path.join(Config.UPLOAD_FOLDER, '{}.zip'.format(sess_id)), 'a') as zf:
        buf = io.BytesIO()
        orig = filename.replace(sess_id, '')
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(buf, image_format(orig))
        zf.writestr(orig, buf.getvalue())


def save_image(done_imgs,
               sess_id):
    """
    """
    for filename, img in done_imgs.items():
        zip_image(filename, sess_id, img)


def face_detection(imgfiles, outfilenames, sess_id, index):
    import tensorflow as tensorflow
    filenames = []
    imgs = []
    dets = []
    base_faces = []
    detector = MTCNN()
    face, det, count = crop_face(imgfiles[index], detector, outfilenames[index], sess_id)
    if face is not None:
        for f, d in zip(face, det):
            img = np.around(imgfiles[index] / 255.0, decimals=12)
            base_faces.append(f)
            imgs.append(img)
            dets.append(d)
            filenames.append(outfilenames[index])

    return base_faces, filenames, imgs, dets, count


def get_filename_dict(filenames, dets):
    filename_dict = {}
    for file in filenames:
        d = dets.item()
        for i, val in d[file].items():
            filename_dict[i] = file
    return filename_dict


def load_images(params, data, pairs):
    """
    """
    print('Loading Images...')
    people = []
    dets = data['dets']
    filenames = data['filenames']
    counts = data['counts']
    selected = data['selected']
    filename_dict = get_filename_dict(filenames, dets)
    print(filename_dict)
    for key, face_matches in pairs.items():
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
            face = read_face_image(file[:index], counts[i])
            img = read_full_image(file)

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
            person['target'] = read_face_from_aligned(temp_files[:8], params)

            person['base']['face'] = np.squeeze(np.array(person['base']['face']))
            if len(person['base']['face'].shape) <= 3:
                person['base']['face'] = np.expand_dims(person['base']['face'], axis=0)
            print(person['base']['face'].shape)
            person['base']['source'] = np.squeeze(np.array(person['base']['source']))
            if len(person['base']['source'].shape) <= 3:
                person['base']['source'] = np.expand_dims(person['base']['source'], axis=0)
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


def _face_recognition(faces, threshold, batch_size):
    with Pool(processes=1) as pool:
        return pool.apply(face_recognition, (faces, threshold, batch_size))


def recognize(data):
    filedets = data['filedets']
    filenames = data['filenames']
    counts = data['counts']
    filename_dict = get_filename_dict(filenames, filedets)
    base_faces = []
    for i, file in filename_dict.items():
        index = file.index('.')
        face = read_face_image(file[:index], counts[i])
        base_faces.append(face)
    faces = np.squeeze(np.array(base_faces))
    if len(base_faces) <= 1:
        faces = np.expand_dims(faces, axis=0)

    tf_config = Config.set_gpu('0')
    embeddings, buckets, means = _face_recognition(faces, 13, 10) # faces should be in an npz
    pairs = match_closest(means)
    lfw_pairs = {}
    for key, val in pairs.items():
        lfw_pairs[key] = {'pair': val, 'faces': buckets[key]}

    return lfw_pairs


def read_face_from_aligned(file_list, params):
    """
    Description

    Keyword arguments:
    """
    result = []
    print(file_list[0])
    for file_name in file_list:
        # print(file_name)
        face = imageio.imread(file_name)
        # print(face)
        face = pre_proc(face, params)
        # print(face)
        result.append(face)
    result = np.array(result)
    return result


def crop_face(img, detector, outfilename, sess_id):
    """
    Description

    Keyword arguments:
    """
    # interpolation = params['interpolation']
    interpolation = cv2.INTER_LINEAR
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    margin = 4
    image_width = 96
    image_height = 112

    print('Trying to find a bounding box')
    try:
        bounding_boxes = detector.detect_faces(img)
        nrof_faces = len(bounding_boxes)
    except:
        print('Error detecting')
        return None, None
    if nrof_faces < 1:
        print('Error, found {} faces'.format(nrof_faces))
        return None, None
    dets = []
    faces = []
    count = 0
    for b in bounding_boxes:
        det = b['box']
        det[2] += det[0]
        det[3] += det[1]
        img_size = np.asarray(img.shape)[0:2]

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = cv2.resize(cropped, (image_height, image_width), interpolation)
        scaled = scaled[...,::-1]

        index = outfilename.index('.')
        orig = '{}_{}.png'.format(outfilename[:index], count)
        zip_image(orig, sess_id, scaled)
        count += 1
        
        face = np.around(np.transpose(scaled, (2,0,1))/255.0, decimals=12)
        face = (face-0.5)*2
        dets.append(det)
        faces.append(face)
    
    faces = np.array(faces)
    print(face.shape)
    
    return faces, dets, count


def read_full_image(file):
    img = imageio.imread(os.path.join(Config.UPLOAD_FOLDER, file))
    img = np.around(img / 255.0, decimals=12)
    return img


def read_face_image(file, count):
    face = imageio.imread(os.path.join(Config.UPLOAD_FOLDER, '{}_{}.png'.format(file, count)))
    face = np.around(np.transpose(face, (2,0,1))/255.0, decimals=12)
    face = (face-0.5)*2
    return face


def apply_delta(delta, img, det, params):
    """
    Description

    Keyword arguments:
    """

    margin = 4
    img_size = np.asarray(img.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])

    orig_dim = [bb[3]-bb[1], bb[2]-bb[0]]

    delta_up = cv2.resize(delta, (orig_dim[1], orig_dim[0]), params['interpolation'])
    print(delta_up.shape)
    print(img.shape)
    img[bb[1]:bb[3],bb[0]:bb[2],:] += delta_up
    img[bb[1]:bb[3],bb[0]:bb[2],:] = np.maximum(img[bb[1]:bb[3],bb[0]:bb[2],:], 0)
    img[bb[1]:bb[3],bb[0]:bb[2],:] = np.minimum(img[bb[1]:bb[3],bb[0]:bb[2],:], 1)

    return img


def pre_proc(img, params):
    """
    Description

    Keyword arguments:
    img -- 
    params -- 
    """
    interpolation = params['interpolation']
    # img: read in by imageio.imread
    # with shape (x,y,3), in the format of RGB
            # convert to (3,112,96) with BGR
    img_resize = cv2.resize(img, (112, 96), interpolation)
    img_BGR = img_resize[...,::-1]
    img_CHW = (img_BGR.transpose(2, 0, 1) - 127.5) / 128
    return img_CHW
