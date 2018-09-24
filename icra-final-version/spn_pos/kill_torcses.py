import os

def kill_torcses():
	s = os.popen('ps aux | grep torcs').read()
	processes = [l.split()[1] for l in s.split('\n') if '_rgs' in l]
	for process in processes:
		os.system('kill ' + process)

if __name__ == '__main__':
	kill_torcses()