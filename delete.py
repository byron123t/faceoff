import os, time, sys


ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')

while True:
	now = time.time()
	for f in os.listdir(UPLOAD_FOLDER):
		path = os.path.join(UPLOAD_FOLDER, f)
		if os.stat(path).st_mtime < now - 3600:
			if os.path.isfile(path):
				print(path)
				os.remove(path)
	time.sleep(1)
