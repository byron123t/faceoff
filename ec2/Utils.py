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
