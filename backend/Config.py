import os, cv2
import numpy as np


ROOT = os.path.abspath('./backend')
ALIGN_96_DIR = 'lfw/lfw-aligned-96'
ALIGN_160_DIR = 'half_celeb160'
VGG_ALIGN_160_DIR = 'small-vgg-align-train-160'
VGG_VALIDATION_DIR = 'small-vgg-align-validate'
TEST_DIR = 'test_imgs'
FULL_DIR = 'half_celeb'
VGG_TEST_DIR = 'test_imgs/VGG'
OUT_DIR = 'new_adv_imgs'
API_DIR = 'new_api_results'

CASIA_MODEL_PATH = 'weights/facenet_casia.h5'
VGGSMALL_MODEL_PATH = 'weights/small_facenet_center.h5'
VGGADV_MODEL_PATH = 'weights/facenet_vggsmall.h5'
CENTER_MODEL_PATH = 'weights/facenet_keras_center_weights.h5'
TRIPLET_MODEL_PATH = 'weights/facenet_keras_weights.h5'
UPLOAD_FOLDER = os.path.join(os.path.abspath('.'), 'static', 'temp')

NAMES = ['barack', 'bill', 'jenn', 'leo', 'mark', 'matt', 'melania', 'meryl',
         'morgan', 'taylor']
API_PEOPLE = ['barack', 'leo', 'matt', 'melania', 'morgan', 'taylor']
PAIRS = {'barack': 'morgan', 'mark': 'bill', 'matt': 'bill', 'taylor': 'jenn',
         'melania': 'jenn', 'jenn': 'melania', 'bill': 'barack', 'morgan':
         'bill', 'leo': 'bill', 'meryl': 'jenn'}


def string_to_bool(arg):
    """Converts a string into a returned boolean."""
    if arg.lower() == 'true':
        arg = True
    elif arg.lower() == 'false':
        arg = False
    else:
        raise ValueError('ValueError: Argument must be either "true" or "false".')
    return arg


def set_gpu(gpu):
    import tensorflow as tf
    """Configures CUDA environment variable and returns tensorflow GPU config."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    return tf_config


def set_parameters(targeted_flag='true',
                   tv_flag='false',
                   hinge_flag='true',
                   cos_flag='false',
                   interpolation='bilinear',
                   model_type='small',
                   loss_type='center',
                   dataset_type='vgg',
                   attack='CW',
                   norm='2',
                   epsilon=0.1,
                   iterations=100,
                   binary_steps=8,
                   learning_rate=0.01,
                   epsilon_steps=0.01,
                   init_const=0.3,
                   mean_loss='embeddingmean',
                   batch_size=-1,
                   margin=5.0,
                   amplification=2.0):
    """Creates and returns a dictionary of parameters."""
    params = {}

    params['model_type'] = model_type
    params['loss_type'] = loss_type
    params['dataset_type'] = dataset_type
    params['attack'] = attack
    params['norm'] = norm
    params['epsilon'] = epsilon
    params['iterations'] = iterations
    params['binary_steps'] = binary_steps
    params['learning_rate'] = learning_rate
    params['epsilon_steps'] = epsilon_steps
    params['init_const'] = init_const
    params['mean_loss'] = mean_loss
    params['batch_size'] = batch_size
    params['targeted_flag'] = string_to_bool(targeted_flag)
    params['tv_flag'] = string_to_bool(tv_flag)
    params['hinge_flag'] = string_to_bool(hinge_flag)
    params['cos_flag'] = string_to_bool(cos_flag)
    params['margin'] = margin
    params['amp'] = amplification

    if model_type == 'small' and loss_type == 'center':
        params['pixel_max'] = 1.0
        params['pixel_min'] = -1.0
    else:
        params['pixel_max'] = 1.0
        params['pixel_min'] = 0.0

    if (dataset_type == 'vggsmall'):
        params['align_dir'] = VGG_ALIGN_160_DIR
        params['test_dir'] = VGG_TEST_DIR
    elif model_type == 'large' or dataset_type == 'casia':
        params['align_dir'] = ALIGN_160_DIR
    elif model_type == 'small':
        params['align_dir'] = ALIGN_96_DIR
    else:
        ValueError('ValueError: Argument must be either "small" or "large".')
    
    if interpolation == 'nearest':
        params['interpolation'] = cv2.INTER_NEAREST
    elif interpolation == 'bilinear':
        params['interpolation'] = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        params['interpolation'] = cv2.INTER_CUBIC
    elif interpolation == 'lanczos':
        params['interpolation'] = cv2.INTER_LANCZOS4
    elif interpolation == 'super':
        print('finish later')
    else:
        raise ValueError('ValueError: Argument must be of the following, [nearest, bilinear, bicubic, lanczos, super].')

    if params['hinge_flag']:
        params['attack_loss'] = 'hinge'
    else:
        params['attack_loss'] = 'target'
    if not params['targeted_flag']:
        params['attack_loss'] = 'target'
    if norm == 'inf':
        norm_name = 'i'
    else:
        norm_name = '2'
    if params['tv_flag']:
        tv_name = '_tv'
    else:
        tv_name = ''
    if params['cos_flag']:
        cos_name = '_cos'
    else:
        cos_name = ''

    params['model_name'] = '{}_{}'.format(model_type, loss_type)
    if dataset_type == 'casia' or dataset_type == 'vggsmall':
        params['model_name'] = dataset_type
    params['attack_name'] = '{}_l{}{}{}'.format(attack.lower(), norm_name, tv_name, cos_name)

    return params
