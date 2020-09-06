from flask import Flask, flash, request, redirect, url_for, render_template, session, json, send_from_directory
import requests
import os
import secrets
import numpy as np
import cv2
from uuid import uuid4
from werkzeug.utils import secure_filename
from ec2.Utils import face_detection, save_image, zip_image
from wtforms import Form, BooleanField, StringField, validators, MultipleFileField, widgets, RadioField, HiddenField, SubmitField
from wtforms.csrf.session import SessionCSRF
from wtforms.fields.html5 import IntegerRangeField
from flask_wtf import FlaskForm, RecaptchaField
from datetime import datetime
from functools import partial
from multiprocessing import Pool, Manager, Value
from zipfile import ZipFile
import redis
import string
import random
import paramiko
from filelock import FileLock
from rq import Queue, Worker, Connection


ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# RECAPTCHA_PUBLIC_KEY = '6LdldMIZAAAAADWxxMHKOlH3mFFxt8BRVJAkSf6T'
# RECAPTCHA_PRIVATE_KEY = '6LdldMIZAAAAAPyqq3ildSIGiPRcBJa-loTmj6vN'
RECAPTCHA_PUBLIC_KEY = '6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI'
RECAPTCHA_PRIVATE_KEY = '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe'

HOST = '10.0.0.24'
PORT = 22
USERNAME = 'asianturtle'
REM_ROOT = '/home/asianturtle/faceoff'


app = Flask(__name__)
r = redis.Redis()
q = Queue(connection=r)
app.config['SECRET_KEY'] = secrets.token_bytes(16)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECAPTCHA_PUBLIC_KEY'] = RECAPTCHA_PUBLIC_KEY
app.config['RECAPTCHA_PRIVATE_KEY'] = RECAPTCHA_PRIVATE_KEY


class DetectFormBase(FlaskForm):
    submit = SubmitField('Submit')


def file_list_form_builder(length):
    class DetectListForm(DetectFormBase):
        pass

    for i in range(length+1):
        setattr(DetectListForm, 'customSwitch{}'.format(i), BooleanField())

    return DetectListForm()


class PhotoForm(FlaskForm):
    photo = MultipleFileField('images')
    recaptcha = RecaptchaField()


class PertForm(FlaskForm):
    perts = IntegerRangeField('perts', default=2)
    attacks = IntegerRangeField('attacks', default=0)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in EXTENSIONS


def _session_attr(key):
    try:
        return session[key]
    except KeyError:
        return None


@app.route('/', methods=['GET', 'POST'])
def handle_upload(name=None):
    form = PhotoForm()
    # sess_id = str(uuid4())
    sess_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))
    with FileLock('ids.txt.lock', timeout=10.0) as fl:
        with open('ids.txt', 'r') as userids:
            txt = userids.read()
            while sess_id in userids:
                # sess_id = str(uuid4())
                sess_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))

    if request.method == 'POST':
        if form.is_submitted():
            with ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], '{}.zip'.format(sess_id)), 'w') as zf:
                files = form.photo.data
                imgfiles = []
                outfilenames = []
                for file in files:
                    filename = secure_filename(file.filename)
                    if file and allowed_file(filename):
                        index = filename.index('.')
                        outfile = filename[:index] + sess_id + filename[index:]
                        outfilenames.append(outfile)
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], outfile))
                        npimg = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], outfile))
                        zip_image(outfile, sess_id, npimg)
                        imgfiles.append(npimg)
                if len(imgfiles) == 0:
                    return redirect(request.url)
        else:
            return redirect(request.url)
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
        session['filedets'] = json.dumps(filedets)
        session['filedims'] = json.dumps(filedims)
        session['filenames'] = json.dumps(filenames)
        session['counts'] = json.dumps(counts)
        session['sess_id'] = sess_id
        return redirect(url_for('detected_faces'))
    else:
        return render_template('index.html', name=name, form=form)


@app.route('/detected_faces', methods=['GET', 'POST'])
def detected_faces():
    count = 0
    filedets = _session_attr('filedets')
    filedims = _session_attr('filedims')

    if filedets is not None and filedims is not None:
        filedets = json.loads(filedets)
        filedims = json.loads(filedims)
    else:
        return redirect(url_for('handle_upload'))
    for key, val in filedets.items():
        count += len(val.keys())
        print(count)
    form = file_list_form_builder(count)
    print(filedets)
    if request.method == 'GET':
        return render_template('detect.html', filedets=filedets, filedims=filedims, form=form)
    else:
        print(request.form)
        for fieldname, value, in request.form.items():
            print(fieldname, value)
        selected = []
        for key, val in request.form.items():
            print(val)
            if val == 'y':
                selected.append(int(key.replace('customSwitch', '')))
        session['selected'] = selected
        return redirect(url_for('select_pert'))


@app.route('/select_pert', methods=['GET', 'POST'])
def select_pert():
    sess_id = _session_attr('sess_id')
    selected = _session_attr('selected')
    filedets = _session_attr('filedets')
    filedims = _session_attr('filedims')
    filenames = _session_attr('filenames')
    counts = _session_attr('counts')
    if sess_id is None or selected is None:
        return redirect(url_for('handle_upload'))
    form = PertForm()
    att_desc = {'desc1': ['Long attack (CWLI)', 'Normal attack (CWL2)', 'Quick attack (PGDL2)'],
                'desc2': ['~10 minutes', '~2 minutes', '~30 seconds']}
    per_desc = {'desc1': ['Heavy Distortion', 'Normal Distortion', 'Light Distortion'],
                'desc2': ['More Privacy', 'Some Privacy', 'Less Privacy']}
    att_file = ['img/logo-dark.png', 'img/logo-dark.png', 'img/logo-dark.png']
    per_file = ['img/logo-dark.png', 'img/logo-dark.png', 'img/logo-dark.png']
    if request.method == 'GET':
        return render_template('pert.html', form=form, att_desc=att_desc, per_desc=per_desc, att_file=att_file, per_file=per_file)
    else:
        print(request.form)
        print(form.perts.data)
        print(form.attacks.data)
        attk = form.attacks.data
        pert = form.perts.data
        if attk == 0:
            attack = 'CW'
            norm = 2
        elif attk == 1:
            attack = 'PGD'
            norm = 2
        else:
            print('error')
        if pert == 0:
            margin = 5
            amplification = 5
        elif pert == 1:
            margin = 5
            amplification = 4.5
        elif pert == 2:
            margin = 5
            amplification = 4
        elif pert == 3:
            margin = 5
            amplification = 3.5
        elif pert == 4:
            margin = 5
            amplification = 3
        else:
            print('error')
        with FileLock('ids.txt.lock', timeout=10.0) as fl:
            with open('ids.txt', 'w') as userids:
                userids.write('{} - {} - {} - {}\n'.format(sess_id, datetime.utcnow(), attk, norm))
        np.savez(os.path.join(app.config['UPLOAD_FOLDER'], '{}data.npz'.format(sess_id)),
                 sess_id=sess_id,
                 filedets=filedets,
                 filedims=filedims,
                 filenames=filenames,
                 counts=counts,
                 margin=margin,
                 amplification=amplification,
                 attack=attack,
                 norm=norm)
        with ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], '{}.zip'.format(sess_id)), 'a') as zf:
            zf.write(os.path.join(app.config['UPLOAD_FOLDER'], '{}data.npz'.format(sess_id)), '{}data.npz'.format(sess_id))

        # try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        k = paramiko.RSAKey.from_private_key_file('/home/asianturtle/.ssh/id_rsa')
        ssh.connect(HOST, username=USERNAME, pkey = k)
        sftp = ssh.open_sftp()
        sftp.chdir(REM_ROOT)
        localfile = os.path.join(app.config['UPLOAD_FOLDER'], '{}.zip'.format(sess_id))
        remotefile = os.path.join(REM_ROOT, 'static', 'temp', '{}.zip'.format(sess_id))
        sftp.put(localfile, remotefile)
        if sftp:
            sftp.close()
        # except Exception as e:
            # print(e)

        return redirect(url_for('download'))


@app.route('/download', methods=['GET', 'POST'])
def download():
    sess_id = _session_attr('sess_id')
    if sess_id is None:
        return redirect(url_for('handle_upload'))
    if request.method == 'GET':
        return send_from_directory(UPLOAD_FOLDER, '{}.zip'.format(sess_id), as_attachment=True)
    else:
        print(request.form)
        return redirect(url_for('handle_upload'))


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')
