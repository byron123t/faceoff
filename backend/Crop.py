import os
import numpy as np
import imageio
import cv2
from backend import Config


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

    try:
        bounding_boxes = detector.detect_faces(img)
        nrof_faces = len(bounding_boxes)
    except:
        return None, None, 0
    if nrof_faces < 1:
        return None, None, 0
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
        imageio.imwrite(os.path.join(Config.UPLOAD_FOLDER, '{}{}_{}.png'.format(sess_id, outfilename[:index], count)),scaled)
        count += 1
        
        face = np.around(np.transpose(scaled, (2,0,1))/255.0, decimals=12)
        face = (face-0.5)*2
        dets.append(det)
        faces.append(face)
    
    faces = np.array(faces)
    
    return faces, dets, count


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
    img[bb[1]:bb[3],bb[0]:bb[2],:3] += delta_up
    img[bb[1]:bb[3],bb[0]:bb[2],:3] = np.maximum(img[bb[1]:bb[3],bb[0]:bb[2],:3], 0)
    img[bb[1]:bb[3],bb[0]:bb[2],:3] = np.minimum(img[bb[1]:bb[3],bb[0]:bb[2],:3], 1)

    return img


def read_face_from_aligned(file_list, params):
    """
    Description

    Keyword arguments:
    """
    result = []
    for file_name in file_list:
        # print(file_name)
        face = imageio.imread(file_name)
        # print(face)
        face = pre_proc(face, params)
        # print(face)
        result.append(face)
    result = np.array(result)
    return result
