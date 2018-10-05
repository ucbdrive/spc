import numpy as np
import cv2

def read_action(fname):
    with open(fname, 'r') as f:
        s = f.readlines()
    action = np.zeros((10, 2))
    action[0] = np.array(eval(s[0][13:-1]))
    for j in range(1, 10):
        l = s[2 + 5 * j]
        k = 0
        while l[k] != '[':
            k += 1
        k += 1
        while l[k] == ' ':
            k += 1
        q = k + 1
        while l[q] != ' ':
            q += 1
        action[j, 0] = float(l[k:q])
        k = q
        while l[k] == ' ':
            k += 1
        q = k + 1
        while l[q] != ']':
            q += 1
        action[j, 1] = float(l[k:q])
    speed = np.zeros(10)
    for j in range(10):
        speed[j] = float(s[5+5*j][10:-2])
    return action, speed

def draw_action(fig, x, y, l, w, action):
    fig[x-l:x+l, y-w:y+w] = 0
    fig[x-w:x+w, y-l:y+l] = 0
    t = int(abs(action[0]) * l)
    if action[0] > 0:
        fig[x-t:x, y-3*w:y+3*w] = np.array([36, 28, 237])
    else:
        fig[x:x+t, y-3*w:y+3*w] = np.array([36, 28, 237])
    t = int(abs(action[1]) * l)
    if action[1] < 0:
        fig[x-3*w:x+3*w, y:y+t] = np.array([14, 201, 255])
    else:
        fig[x-3*w:x+3*w, y-t:y] = np.array([14, 201, 255])
    return fig

fr = {'base': 54, 'forward': 60, 'right': 24}
name = 'right'

video = cv2.VideoWriter('%s.avi' % name, cv2.VideoWriter_fourcc(*'MJPG'), fr[name], (1920, 1080), True)

with open('full/act_%s.txt' % name, 'r') as f:
    s = f.readlines()
for i in range(len(s)):
    s[i] = s[i].split(' ')
    s[i][0] = int(float(s[i][0]) * fr[name])
cap = cv2.VideoCapture('full/full_%s.avi' % name)
cap.set(1, s[0][0])
t = 0

for i in range(20*fr[name]-s[0][0]):
    ret, obs = cap.read()
    # frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 25
    # obs = cv2.resize(obs, (1370, 1080))
    # frame[:, :1370] = obs
    frame = cv2.resize(obs, (1920, 1080))

    if t < len(s) and i + s[0][0] == s[t][0]:
        action, speed = read_action('full/%s/%d/outcome/log.txt' % (name, t))
        # segs = []
        # for j in range(10):
        #     seg = cv2.resize(cv2.imread('full/%s/%d/outcome/seg%d.png' % (name, t, j+1)), (265, 208))
        #     seg = draw_action(seg, 150, 190, 25, 1, action[j])
        #     cv2.putText(seg, 'Step %d' % (j + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 248, 246), 2)
        #     cv2.putText(seg, 'speed: %0.2f' % (speed[j]), (10, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (248, 248, 246), 2)
        #     segs.append(seg)
        t += 1
    # for j in range(10):
    #     frame[(int(j/2)*218):(int(j/2)*218+208), (int(j%2)*275+1380):(int(j%2)*275+1380+265)] = segs[j]
    frame = draw_action(frame, 780, 1500, 160, 2, action[0])
    video.write(frame)

video.release()