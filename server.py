from flask import Flask, flash, request, redirect, url_for, render_template, session, json, send_from_directory
import requests
import os
import secrets
import numpy as np
import cv2
from uuid import uuid4
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from faceoff.Utils import face_detection, face_recognition, match_closest, load_images, save_image
from faceoff import Config
from faceoff.listener import attack_listener, recognize_listener
from faceoff.Detect import detect_listener
from faceoff.Attack import amplify
from wtforms import Form, BooleanField, StringField, validators, MultipleFileField, widgets, RadioField, HiddenField, SubmitField
from wtforms.csrf.session import SessionCSRF
from wtforms.fields.html5 import IntegerRangeField
from flask_wtf import FlaskForm, RecaptchaField
from functools import partial
from multiprocessing import Pool, Manager, Value
from zipfile import ZipFile
from redis import Redis
import string
import random
import time
from datetime import datetime
from filelock import FileLock
from rq import Queue, Worker, Connection, Retry


ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# RECAPTCHA_PUBLIC_KEY = '6LdldMIZAAAAADWxxMHKOlH3mFFxt8BRVJAkSf6T'
# RECAPTCHA_PRIVATE_KEY = '6LdldMIZAAAAAPyqq3ildSIGiPRcBJa-loTmj6vN'
RECAPTCHA_PUBLIC_KEY = '6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI'
RECAPTCHA_PRIVATE_KEY = '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe'


app = Flask(__name__)
gpu = Queue('gpu', connection=Redis())
cpu = Queue('cpu', connection=Redis())
app.config['SECRET_KEY'] = secrets.token_bytes(16)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECAPTCHA_PUBLIC_KEY'] = RECAPTCHA_PUBLIC_KEY
app.config['RECAPTCHA_PRIVATE_KEY'] = RECAPTCHA_PRIVATE_KEY
app.config['TRAP_HTTP_EXCEPTIONS']=True


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


class BlankForm(FlaskForm):
    placeholder = StringField('placeholder', default='')


class DownloadForm(FlaskForm):
    download = StringField('download', default='')


class LoopForm(FlaskForm):
    doagain = StringField('doagain', default='')


class PertForm(FlaskForm):
    perts = IntegerRangeField('perts', default=2)
    attacks = IntegerRangeField('attacks', default=0)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in EXTENSIONS


def session_attr(key):
    try:
        return session[key]
    except KeyError:
        return None


def error_message(message):
    session['error'] = message
    return redirect(url_for('handle_upload'))


@app.errorhandler(RequestEntityTooLarge)
def handle_bad_request(e):
    return error_message('Uploaded files must be under 5MB')


@app.route('/', methods=['GET', 'POST'])
def handle_upload():
    form = PhotoForm()
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return render_template('index.html', form=form, error=err)
    if request.method == 'POST':
        sess_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))
        try:
            with FileLock('ids.txt.lock', timeout=10.0) as fl:
                with open('ids.txt', 'r') as userids:
                    txt = userids.read()
                    while sess_id in userids:
                        # sess_id = str(uuid4())
                        sess_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))
                with open('ids.txt', 'a') as userids:
                    userids.write('{}\n'.format(sess_id))
        except Exception as e:
            print(e)
            os.remove('ids.txt.lock')
        if form.is_submitted():
            files = form.photo.data
            outfilenames = []
            for file in files:
                filename = secure_filename(file.filename)
                if file and allowed_file(filename):
                    index = filename.index('.')
                    outfile = filename[:index] + sess_id + filename[index:]
                    outfilenames.append(outfile)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], outfile))

            if len(outfilenames) == 0:
                return error_message('Please upload at least 1 file of the following type [.jpeg .jpg .png .gif]')
        else:
            return redirect(request.url)
        session['outfilenames'] = outfilenames
        session['sess_id'] = sess_id
        return redirect(url_for('uploading'))
    else:
        return render_template('index.html', form=form)


@app.route('/uploading', methods=['GET', 'POST'])
def uploading():
    sess_id = session_attr('sess_id')
    outfilenames = session_attr('outfilenames')
    if sess_id is None:
        return error_message('Sorry, your session has expired.')
    form = BlankForm()
    if request.method == 'POST':
        try:
            with FileLock('ids.txt.lock', timeout=10.0) as fl:
                with open('ids.txt', 'r') as userids:
                    text = userids.read()
                    if '{}loading'.format(sess_id) in text:
                        return redirect(url_for('detected_faces'))
                    new_text = text.replace(sess_id, sess_id+'loading')
                with open('ids.txt', 'w') as userids:
                    userids.write(new_text)
        except Exception as e:
            print(e)
            os.remove('ids.txt.lock')
        imgfiles = []
        for outfile in outfilenames:
            npimg = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], outfile))
            imgfiles.append(npimg)
        jobs = []
        base_faces = []
        filenames = []
        imgs = []
        img_map = {}
        dets = []
        counts = []
        for i in range(len(imgfiles)):
            jobs.append(cpu.enqueue(detect_listener, imgfiles[i], outfilenames[i], sess_id, job_timeout=15, retry=Retry(max=10, interval=1)))
        for i, job in enumerate(jobs):
            while job.result is None:
                time.sleep(1)
            b, f, im, d, c = job.result
            base_faces.extend(b)
            filenames.extend(f)
            imgs.append(im)
            dets.extend(d)
            for j in range(c):
                counts.append(j)
                img_map[j] = i
        job = gpu.enqueue(recognize_listener, base_faces, filenames, dets, imgs, counts, img_map, sess_id)
        while job.result is None:
            time.sleep(1)
        filedets, filedims = job.result
        session['filedets'] = json.dumps(filedets)
        session['filedims'] = json.dumps(filedims)
        return redirect(url_for('detected_faces'))
    else:
        return render_template('uploading.html', form=form)


@app.route('/detected_faces', methods=['GET', 'POST'])
def detected_faces():
    count = 0
    sess_id = session_attr('sess_id')
    filedets = session_attr('filedets')
    filedims = session_attr('filedims')

    if sess_id and filedets and filedims:
        filedets = json.loads(filedets)
        filedims = json.loads(filedims)
    else:
        return error_message('Sorry, your session has expired.')
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
    sess_id = session_attr('sess_id')
    selected = session_attr('selected')
    if sess_id is None or selected is None:
        return error_message('Sorry, your session has expired.')
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
        session['attack'] = form.attacks.data
        session['pert'] = form.perts.data
        return redirect(url_for('download'))


@app.route('/download', methods=['GET', 'POST'])
def download():
    sess_id = session_attr('sess_id')
    if sess_id is None:
        return error_message('Sorry, your session has expired.')
    form = BlankForm()
    if request.method == 'GET':
        return render_template('download.html', form=form, sess_id=sess_id)
    else:
        attack, norm, margin, amplification, selected = parse_form()
        try:
            with FileLock('ids.txt.lock', timeout=10.0) as fl:
                with open('ids.txt', 'r') as userids:
                    text = userids.read()
                    print(text)
                    print('{} - {} - {}'.format(sess_id, attack, norm))
                    if '{} - {} - {}'.format(sess_id, attack, norm) in text:
                        return redirect(url_for('finish'))
                    new_text = text.replace(sess_id+'loading', '{} - {} - {} - {}'.format(sess_id, attack, norm, datetime.utcnow()))
                with open('ids.txt', 'w') as userids:
                    userids.write(new_text)
        except Exception as e:
            print(e)
            os.remove('ids.txt.lock')
        params, people = pre_proc_attack(attack, norm, margin, amplification, selected, sess_id)
        jobs = []
        done_imgs = {}
        for i in range(len(people)):
            jobs.append(gpu.enqueue(attack_listener, params, people[i], sess_id, job_timeout=1500, retry=Retry(max=10, interval=1)))
        for i in jobs:
            while i.result is None:
                time.sleep(1)
            person, delta = i.result
            done_imgs = amplify(params=params,
                                face=person['base']['face'],
                                delta=delta,
                                amp=params['amp'],
                                person=person,
                                done_imgs=done_imgs)
        save_image(done_imgs=done_imgs,
               sess_id=sess_id)
        return redirect(url_for('finish'))


@app.route('/finish', methods=['GET', 'POST'])
def finish():
    sess_id = session_attr('sess_id')
    if sess_id is None:
        return error_message('Sorry, your session has expired.')
    form1 = DownloadForm()
    form2 = LoopForm()
    if request.method == 'GET':
        return render_template('finish.html', form1=form1, form2=form2, sess_id=sess_id)
    else:
        print(form1.data)
        print(form2.data)
        if form1.data['download'] == 'yes':
            return send_from_directory(UPLOAD_FOLDER, '{}.zip'.format(sess_id), as_attachment=True)
        elif form2.data['doagain'] == 'yes':
            return redirect(url_for('handle_upload'))


def parse_form():
    attk = session_attr('attack')
    pert = session_attr('pert')
    selected = session_attr('selected')
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
        amplification = 4.75
    elif pert == 1:
        margin = 5
        amplification = 4.00
    elif pert == 2:
        margin = 5
        amplification = 3.25
    elif pert == 3:
        margin = 5
        amplification = 2.50
    elif pert == 4:
        margin = 5
        amplification = 1.75
    else:
        print('error')
    return attack, norm, margin, amplification, selected


def pre_proc_attack(attack, norm, margin, amplification, selected, sess_id):
    params = Config.set_parameters(attack=attack,
                                   norm=norm,
                                   margin=margin,
                                   amplification=amplification,
                                   mean_loss='embedding')
    people = load_images(params=params,
                         selected=selected,
                         sess_id=sess_id)
    return params, people


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')
