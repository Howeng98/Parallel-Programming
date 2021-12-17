from subprocess import Popen, PIPE, STDOUT

p = Popen(["rm /tmp/.nvprof/*"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)