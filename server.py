from flask import Flask, flash, request, redirect, url_for, render_template
import requests
import os
import secrets
from werkzeug.utils import secure_filename

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
        for file in files:
            filename = secure_filename(file.filename)
            if file and allowed_file(filename):
                print(os.path.join(app.config['UPLOAD_FOLDER'], filename), flush=True)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('handle_upload'))