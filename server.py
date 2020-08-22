from flask import Flask, flash, request, redirect, url_for, render_template, session, json
import requests
import os
import secrets
import numpy as np
import cv2
from uuid import uuid4
from werkzeug.utils import secure_filename
from faceoff.Utils import face_detection, face_recognition, match_closest, load_images
from faceoff import Config
from faceoff.Attack import outer_attack
from wtforms import Form, BooleanField, StringField, validators, MultipleFileField, widgets, RadioField, HiddenField, SubmitField
from wtforms.csrf.session import SessionCSRF
from flask_wtf import FlaskForm
from datetime import timedelta


ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_bytes(16)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



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


class PertForm(FlaskForm):
    perts = RadioField('perts', choices=['more', 'some', 'less'])
    attacks = RadioField('attacks', choices=['cwli', 'cwl2', 'pgdl2'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def handle_upload(name=None):
    form = PhotoForm()
    if request.method == 'POST':
        if form.is_submitted():
            print(form.photo.data)
            files = form.photo.data
            imgfiles = []
            outfilenames = []
            for file in files:
                filename = secure_filename(file.filename)
                if file and allowed_file(filename):
                    index = filename.index('.')
                    sess_id = str(uuid4())
                    outfile = filename[:index] + sess_id + filename[index:]
                    outfilenames.append(outfile)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], outfile))
                    npimg = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], outfile))
                    print(npimg.shape)
                    imgfiles.append(npimg)
        else:
            print('No selected file', flush=True)
            flash('No selected file')
            return redirect(request.url)

        tf_config = Config.set_gpu('0')
        faces, filenames, imgs, dets = face_detection(imgfiles, outfilenames)
        print(faces.shape)
        print(len(dets))
        print(len(imgs))
        embeddings, buckets, means = face_recognition(faces, 15.463615, 10, tf_config)
        pairs = match_closest(means)
        lfw_pairs = {}
        for key, val in pairs.items():
            lfw_pairs[val] = buckets[key]
            lfw_pairs[val].append(key)
        print(pairs)
        print(buckets)
        filedets = {}
        filedims = {}
        count = 0
        for f, d, i in zip(filenames, dets, imgs):
            if f not in filedets:
                filedets[f] = {}
                filedims[f] = {}
            filedims[f]['height'] = i.shape[0]
            filedims[f]['width'] = i.shape[1]
            filedets[f][count] = d.tolist()
            count += 1
        print(filedets)
        print(filedims)
        np.savez(os.path.join(app.config['UPLOAD_FOLDER'], sess_id + 'data.npz'), pairs=lfw_pairs, dets=filedets, filenames=filenames)
        return redirect(url_for('detected_faces', filedets=json.dumps(filedets), filedims=json.dumps(filedims), sess_id=sess_id))
    else:
        return render_template('index.html', name=name, form=form)


@app.route('/detected_faces/<filedets>/<filedims>/<sess_id>', methods=['GET', 'POST'])
def detected_faces(filedets=None, filedims=None, sess_id=None):
    if filedets is not None:
        filedets = json.loads(filedets)
    if filedims is not None:
        filedims = json.loads(filedims)
    for key, val in filedets.items():
        count = int(max(val.keys()))
        print(count)
    form = file_list_form_builder(count)
    if request.method == 'GET':
        return render_template('detect.html', filedets=filedets, filedims=filedims, form=form)
    else:
        print(request.form)
        for fieldname, value, in request.form.items():
            print(fieldname, value)
        selected = []
        for key, val in request.form.items():
            if val == 'on':
                selected.append(key.replace('customSwitch', ''))
        return redirect(url_for('select_pert', sess_id=sess_id, selected=selected))


@app.route('/select_pert/<sess_id>/<selected>', methods=['GET', 'POST'])
def select_pert(sess_id=None, selected=None):
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
        if attk == 'cwl2':
            attack = 'CW'
            norm = 2
        elif attk == 'cwli':
            attack = 'CW'
            norm = 'inf'
        elif attk == 'pgdl2':
            attack = 'PGD'
            norm = 2
        else:
            print('error')
        if pert == 'more':
            margin = 5
            amplification = 4
        elif pert == 'some':
            margin = 3
            amplification = 2
        elif pert == 'less':
            margin = 2
            amplification = 1.5
        else:
            print('error')
        params = Config.set_parameters(attack=attack,
                                       norm=norm,
                                       margin=margin,
                                       amplification=amplification)
        load_images(selected)
        outer_attack(params=params,
                     )
        return redirect(url_for('download', sess_id=sess_id))


@app.route('/download/<sess_id>', methods=['GET', 'POST'])
def download(sess_id=None):
    if request.method == 'GET':
        return render_template('download.html')
    else:
        print(request.form)
        return redirect(url_for(''))