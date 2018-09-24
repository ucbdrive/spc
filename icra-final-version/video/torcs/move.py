import os
from shutil import copy

# if not os.path.exists('tmp'):
#     os.mkdir('tmp')
# for i in range(1000):
#     copy('%d/outcome/log.txt' % i, 'tmp/%d.txt' % i)
for i in range(1000):
    copy('tmp/%d.txt' % i, '%d/outcome/log.txt' % i)