from flask import Flask, flash, request, redirect, url_for, render_template, session
import requests
import os
import secrets
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from faceoff.Utils import face_detection, face_recognition, match_closest
from faceoff import Config

ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(32)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in EXTENSIONS

@app.route('/')
@app.route('/<name>')
def hello_world(name=None):
    return render_template('index.html', name=name)

@app.route('/', methods=['GET', 'POST'])
def handle_upload(name=None):
    print(session)
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('No file part', flush=True)
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file')
        if len(files) == 0:
            print('No selected file', flush=True)
            flash('No selected file')
            return redirect(request.url)
        imgfiles = []
        for file in files:
            filename = secure_filename(file.filename)
            if file and allowed_file(filename):
                print(os.path.join(app.config['UPLOAD_FOLDER'], filename), flush=True)
                npimg = np.fromstring(file.read(), np.uint8)
                npimg = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
                print(npimg.shape)
                imgfiles.append(npimg)
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        tf_config = Config.set_gpu('0')
        faces, imgs, dets = face_detection(imgfiles)
        print(faces.shape)
        print(len(dets))
        print(len(imgs))
        face_recognition(faces, 17.25696245, 1, tf_config)

        return redirect(url_for('handle_upload'))