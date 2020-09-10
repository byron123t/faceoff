import Config
import time
import paramiko
from datetime import datetime


r = Config.R
HOST = 10.0.0.24
USERNAME = 'asianturtle'
REM_ROOT = '/home/asianturtle'


while True:
	try:
		r.bgsave()
	except Exception as e:
		print(e)
		r = Config.R
	try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        k = paramiko.RSAKey.from_private_key_file('/home/ubuntu/.ssh/id_rsa')
        ssh.connect(HOST, username=USERNAME, pkey = k)
        sftp = ssh.open_sftp()
        sftp.chdir(REM_ROOT)
        localfile = 'dump.rdb'
        remotefile = os.path.join(REM_ROOT, 'dump{}.rdb'.format(datetime.utcnow().strftime('%m-%d-%Y-%H-%M-%S')))
        sftp.put(localfile, remotefile)
        if sftp:
            sftp.close()
	except Exception as e:
		print(e)

	time.sleep(3600)