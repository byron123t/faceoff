from flask import Flask, request, redirect, url_for, render_template, session, json, send_from_directory, Response
import os
import cv2
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from backend.Utils import load_images, save_image
from backend import Config
from backend.listener import attack_listener, recognize_listener
from backend.Detect import detect_listener
from backend.Attack import amplify
from wtforms import BooleanField, StringField, MultipleFileField, SubmitField
from wtforms.fields.html5 import IntegerRangeField
from flask_wtf import FlaskForm, RecaptchaField, validators
from zipfile import ZipFile
from redis import Redis, StrictRedis
import string
import random
import time
import threading
from datetime import datetime, timedelta
from filelock import FileLock
from rq import Queue, Retry


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
r = Config.R
app.config['SECRET_KEY'] = b'\xcd u\xd5\xb4f 5\x18e\x1b\x0f\xf8\xee\x97\xd3'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
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


class AutoForm(FlaskForm):
    auto = StringField('auto', default='')


class DoneForm(FlaskForm):
    done = StringField('done', default='')


class DownloadForm(FlaskForm):
    download = StringField('download', default='')


class LoopForm(FlaskForm):
    doagain = StringField('doagain', default='')


class PertForm(FlaskForm):
    perts = IntegerRangeField('perts', default=2)
    attacks = IntegerRangeField('attacks', default=0)


class DetectThread(threading.Thread):
    def __init__(self, sess_id, outfilenames):
        self.base_faces = []
        self.filenames = []
        self.imgs = []
        self.img_map = {}
        self.dets = []
        self.counts = []
        self.count = 0
        self.progress = 0
        self.sess_id = sess_id
        self.outfilenames = outfilenames
        super().__init__()

    def run(self):
        imgfiles = []
        for outfile in self.outfilenames:
            npimg = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], outfile))
            imgfiles.append(npimg)
        jobs = []
        for i in range(len(imgfiles)):
            jobs.append(cpu.enqueue(detect_listener, imgfiles[i], self.outfilenames[i], self.sess_id, job_timeout=15, retry=Retry(max=10, interval=1)))
        data = r.hgetall(self.sess_id)
        data['progress'] = 0
        r.hset(self.sess_id, mapping=data)

        for i, job in enumerate(jobs):
            while job.result is None:
                time.sleep(1)
            b, f, im, d, c = job.result
            self.base_faces.extend(b)
            self.filenames.extend(f)
            self.imgs.append(im)
            self.dets.extend(d)
            for j in range(c):
                self.counts.append(j)
                self.img_map[self.count] = i
                self.count += 1
            self.progress += 100/(len(jobs)+1)
            print(self.progress)
            data = r.hgetall(self.sess_id)
            data['progress'] = self.progress
            r.hset(self.sess_id, mapping=data)
        job = gpu.enqueue(recognize_listener, self.base_faces, self.filenames, self.dets, self.imgs, self.counts, self.img_map, self.sess_id)
        while job.result is None:
            time.sleep(1)
        data = r.hgetall(self.sess_id)
        data['progress'] = 100
        r.hset(self.sess_id, mapping=data)

        filedets, filedims = job.result
        r.hset(self.sess_id, key='filedets', value=json.dumps(filedets))
        r.hset(self.sess_id, key='filedims', value=json.dumps(filedims))


class AttackThread(threading.Thread):
    def __init__(self, sess_id, attack, margin, amplification, selected):
        self.sess_id = sess_id
        self.attack = attack
        self.margin = margin
        self.amplification = amplification
        self.selected = selected
        self.progress = 0
        super().__init__()

    def run(self):
        data = r.hgetall(self.sess_id)
        data['progress'] = 0
        r.hset(self.sess_id, mapping=data)
        params, people = pre_proc_attack(self.attack, self.margin, self.amplification, self.selected, self.sess_id)
        jobs = []
        done_imgs = {}
        for i in range(len(people)):
            jobs.append(gpu.enqueue(attack_listener, params, people[i], self.sess_id, job_timeout=1500, retry=Retry(max=10, interval=1)))
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
            self.progress += 100/(len(jobs))
            data = r.hgetall(self.sess_id)
            data['progress'] = self.progress
            r.hset(self.sess_id, mapping=data)
        save_image(done_imgs=done_imgs,
               sess_id=self.sess_id)
        data = r.hgetall(self.sess_id)
        data['progress'] = 100
        r.hset(self.sess_id, mapping=data)

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


@app.route('/progress/<sess_id>')
def progress(sess_id):
    def generate():
        data = 0
        while data <= 100:
            data = r.hget(sess_id, 'progress')
            if data is None:
                data = 0
            data = float(data)
            time.sleep(1)
            yield 'data:' + str(data) + '\n\n'
            if data == 100:
                r.hset(sess_id, 'progress', 0)
                break
    return Response(generate(), mimetype='text/event-stream')


@app.route('/', methods=['GET', 'POST'])
def handle_upload():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=30)
    form = PhotoForm()
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return render_template('index.html', form=form, error=err)
    if request.method == 'POST':
        # try:
        #     with FileLock('ids.txt.lock', timeout=10.0) as fl:
        #         with open('ids.txt', 'r') as userids:
        #             txt = userids.read()
        #             while sess_id in userids:
        #                 # sess_id = str(uuid4())
                        
        #         with open('ids.txt', 'a') as userids:
        #             userids.write('{}\n'.format(sess_id))
        # except Exception as e:
        #     print(e)
        #     os.remove('ids.txt.lock')
        if form.validate_on_submit():
            files = form.photo.data
            sess_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))
            while r.exists(sess_id):
                sess_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6))
            data = {'loading': 'false', 'attack': 'none', 'time': 'none', 'amp': 'none', 'margin': 'none', 'progress': 0, 'single': 'none'}
            r.hset(sess_id, mapping=data)
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
            elif len(outfilenames) == 1:
                r.hset(sess_id, 'single', outfilenames[0])
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
    form1 = AutoForm()
    form2 = DoneForm()
    if request.method == 'POST':
        print(form1.data['auto'])
        print(form2.data['done'])
        if form1.data['auto'] == 'yes':
            data = r.hgetall(sess_id)
            print(data['loading'])
            if data['loading'] == 'true':
                return redirect(url_for('detected_faces'))
            data['loading'] = 'true'
            r.hset(sess_id, mapping=data)
            # try:
            #     with FileLock('ids.txt.lock', timeout=10.0) as fl:
            #         with open('ids.txt', 'r') as userids:
            #             text = userids.read()
            #             if '{}loading'.format(sess_id) in text:
            #                 return redirect(url_for('detected_faces'))
            #             new_text = text.replace(sess_id, sess_id+'loading')
            #         with open('ids.txt', 'w') as userids:
            #             userids.write(new_text)
            # except Exception as e:
            #     print(e)
            #     os.remove('ids.txt.lock')
            th = DetectThread(sess_id, outfilenames)
            th.start()
            return ('', 204)
        elif form2.data['done'] == 'yes':
            return redirect(url_for('detected_faces'))
    else:
        return render_template('uploading.html', form1=form1, form2=form2, sess_id=sess_id)


@app.route('/detected_faces', methods=['GET', 'POST'])
def detected_faces():
    count = 0
    sess_id = session_attr('sess_id')

    if sess_id:
        filedets = json.loads(r.hget(sess_id, 'filedets'))
        filedims = json.loads(r.hget(sess_id, 'filedims'))
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
    form1 = AutoForm()
    form2 = DoneForm()
    if request.method == 'GET':
        return render_template('download.html', form1=form1, form2=form2, sess_id=sess_id)
    else:
        print(form1.data['auto'])
        print(form2.data['done'])
        if form1.data['auto'] == 'yes':
            attack, margin, amplification, selected = parse_form()
            data = r.hgetall(sess_id)
            print(data['loading'])
            if data['loading'] == 'false':
                return redirect(url_for('finish'))
            data['attack'] = attack
            data['amp'] = amplification
            data['margin'] = margin
            data['time'] = datetime.utcnow().strftime('%m/%d/%Y, %H:%M:%S')
            r.hmset(sess_id, mapping=data)
            th = AttackThread(sess_id, attack, margin, amplification, selected)
            th.start()
            return ('', 204)
        elif form2.data['done'] == 'yes':
            return redirect(url_for('finish'))


@app.route('/finish', methods=['GET', 'POST'])
def finish():
    sess_id = session_attr('sess_id')
    if sess_id is None:
        return error_message('Sorry, your session has expired.')
    data = r.hgetall(sess_id)
    data['loading'] = 'false'
    r.hset(sess_id, mapping=data)
    form1 = DownloadForm()
    form2 = LoopForm()
    if request.method == 'GET':
        return render_template('finish.html', form1=form1, form2=form2, sess_id=sess_id)
    else:
        if form1.data['download'] == 'yes':
            print(sess_id)
            single = r.hget(sess_id, 'single')
            if single != 'none':
                return send_from_directory(UPLOAD_FOLDER, single, as_attachment=True, attachment_filename=single.replace(sess_id, ''))
            else:
                return send_from_directory(UPLOAD_FOLDER, '{}.zip'.format(sess_id), as_attachment=True)
        elif form2.data['doagain'] == 'yes':
            return redirect(url_for('handle_upload'))


def parse_form():
    attk = session_attr('attack')
    pert = session_attr('pert')
    selected = session_attr('selected')
    if attk == 0:
        attack = 'CW'
        if pert == 0:
            margin = 5
            amplification = 4.00
        elif pert == 1:
            margin = 5
            amplification = 3.25
        elif pert == 2:
            margin = 5
            amplification = 2.50
        elif pert == 3:
            margin = 5
            amplification = 1.75
        elif pert == 4:
            margin = 5
            amplification = 1.00
        else:
            print('error')
    elif attk == 1:
        attack = 'PGD'
        if pert == 0:
            margin = 5
            amplification = 8
        elif pert == 1:
            margin = 5
            amplification = 6.5
        elif pert == 2:
            margin = 5
            amplification = 5
        elif pert == 3:
            margin = 5
            amplification = 3.5
        elif pert == 4:
            margin = 5
            amplification = 2
        else:
            print('error')
    else:
        print('error')

    return attack, margin, amplification, selected


def pre_proc_attack(attack, margin, amplification, selected, sess_id):
    params = Config.set_parameters(attack=attack,
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


if __name__ == '__main__':
    app.run(host='0.0.0.0')
