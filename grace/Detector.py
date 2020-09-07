import paramiko
import os
import imageio
from filelock import FileLock
from zipfile import ZipFile
from grace import Config
from multiprocessing import Pool
from grace.Utils import face_detection


HOST = '169.254.248.203'
PORT = 22
USERNAME = 'byron123t'
REM_ROOT = '/mnt/c/Users/byron.LAPTOP-6A9A5QNU/Desktop/GitHub/faceoff'

INPATH = 'temp/indet'
OUTPATH = 'temp/outdet'
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
        with Pool(processes=3) as pool:
            func = partial(face_detection, imgfiles, outfilenames, sess_id)
            results = pool.map(func, (range(len(imgfiles))))
        base_faces = []
        filenames = []
        imgs = []
        dets = []
        counts = []
        for i, val in enumerate(results):
            base_faces.extend(val[0])
            filenames.extend(val[1])
            imgs.extend(val[2])
            dets.extend(val[3])
            for j in range(val[4]):
                counts.append(j)

        faces = np.squeeze(np.array(base_faces))
        if len(imgs) <= 1:
            faces = np.expand_dims(faces, axis=0)
        # save face np or images
        filedets = {}
        filedims = {}
        count = 0
        for f, d, i in zip(filenames, dets, imgs):
            if f not in filedets:
                filedets[f] = {}
                filedims[f] = {}
            filedims[f]['height'] = i.shape[0]
            filedims[f]['width'] = i.shape[1]
            print(d)
            filedets[f][count] = d
            count += 1
        np.savez(filename.replace('.zip', '.npz'), 
                 filedets=filedets,
                 filedims=filedims,
                 filenames=filenames,
                 counts=counts)
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            k = paramiko.RSAKey.from_private_key_file('/home/asianturtle/.ssh/id_rsa')
            ssh.connect(HOST, username=USERNAME, pkey = k)
            sftp = ssh.open_sftp()
            sftp.chdir(REM_ROOT)
            localfile = os.path.join(INPATH, '{}.npz'.format(sess_id))
            remotefile = os.path.join(REM_ROOT, 'static', 'temp', '{}.zip'.format(sess_id))
            sftp.put(localfile, remotefile)
            if sftp:
                sftp.close()
        except Exception as e:
            print(e)