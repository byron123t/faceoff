import paramiko
import os
import imageio
from filelock import FileLock
from zipfile import ZipFile
from grace import Config
from multiprocessing import Pool
from grace.Attack import outer_attack


HOST = '169.254.248.203'
PORT = 22
USERNAME = 'byron123t'
REM_ROOT = '/mnt/c/Users/byron.LAPTOP-6A9A5QNU/Desktop/GitHub/faceoff'

INPATH = 'temp/infinal'
MIDPATH = 'temp/midfinal'
OUTPATH = 'temp/outfinal'
tf_config = Config.set_gpu('0')
while True:
    files = os.listdir(INPATH)
    if files:
        filename = os.path.join(INPATH, files[0])
        fl = FileLock(filename + '.lock', timeout=0)
        try:
            fl.acquire()
        except FileLockException:
            continue
        with ZipFile(filename, 'r') as zf:
            dir_path = os.path.join(MIDPATH, filename.replace('.zip'))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            ZipFile.extractall(dir_path)
        os.remove(filename)
        fl.release()
        for i, f in os.listdir(dir_path):
            if f.endswith('.npz'):
                data = np.load(os.path.join(zipfolder, f), allow_pickle=True)
        lfw_pairs = recognize(data)
        params = Config.set_parameters(attack=data['attack'],
                                       norm=data['norm'],
                                       margin=data['margin'],
                                       amplification=data['amplification'])
        people = load_images(params=params,
                             data=data,
                             pairs=lfw_pairs)
        with Pool(processes=2) as pool:
            func = partial(outer_attack, params, people)
            results = pool.map(func, (range(len(people))))
        done_imgs = {}
        print('finished!')
        for i, val in enumerate(results):
            person = val[0]
            delta = val[1]
            done_imgs = amplify(params=params,
                                face=person['base']['face'],
                                delta=delta,
                                amp=params['amp'],
                                person=person,
                                done_imgs=done_imgs)
        save_image(done_imgs=done_imgs,
                   sess_id=sess_id)
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            k = paramiko.RSAKey.from_private_key_file('~/.ssh/id_rsa')
            ssh.connect(HOST, username=USERNAME, pkey = k)
            sftp = ssh.open_sftp()
            sftp.chdir(REM_ROOT)
            localfile = os.path.join(Config.UPLOAD_FOLDER, '{}.zip'.format(sess_id))
            remotefile = os.path.join(REM_ROOT, 'static', 'temp', '{}.zip'.format(sess_id))
            sftp.put(localfile, remotefile)
            if sftp:
                sftp.close()
        except Exception as e:
            print(e)