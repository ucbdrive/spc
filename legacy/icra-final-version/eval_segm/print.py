import numpy as np

HIS = 40

def read(fname, his=HIS):
	with open(fname, 'r') as f:
		s = f.readlines()[-his:]
	x = np.zeros((11, his))
	for i in range(his):
		l = s[i].split(' ')
		for j in range(11):
			x[j, i] = float(l[j])
	x = np.mean(x, axis=1)
	return x[1:]

def read10(fname, his=HIS):
	with open(fname, 'r') as f:
		s = f.readlines()[-his:]
	x = np.zeros((10, his))
	for i in range(his):
		l = s[i].split(' ')
		for j in range(10):
			x[j, i] = float(l[j])
	x = np.mean(x, axis=1)
	return x

def main():
	pa = read('pa_log.txt')
	ma = read('accuracy_log.txt')
	miu = read('seg_log.txt')
	fwiu = read('fiu_log.txt')
	coll = read10('coll_log.txt')
	off = read10('off_log.txt')
	with open('out.txt', 'w') as f:
		for i in range(10):
			f.write('%d & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f \\\\\n' % (i+1, pa[i], ma[i], miu[i], fwiu[i], coll[i], off[i]))

if __name__ == '__main__':
	main()