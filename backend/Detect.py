from backend.Utils import face_detection
from multiprocessing import Pool


def _face_detection(imgfiles, outfilenames, sess_id):
    with Pool(processes=1) as pool:
        return pool.apply(face_detection, (imgfiles, outfilenames, sess_id))


def detect_listener(imgfiles, outfilenames, sess_id):
    return _face_detection(imgfiles, outfilenames, sess_id)
