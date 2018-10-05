import numpy as np
import cv2

# seg = cv2.resize(cv2.imread('demo/%d/outcome/seg%d.png' % (0, 1)), (265, 208))
# cv2.putText(seg, 'Step 1', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 248, 246), 2)
# cv2.imwrite('seg_test.png', seg)

# title = np.ones((1080, 1920, 3), dtype=np.uint8) * np.array([[[40, 41, 35]]])
# cv2.putText(title, 'TORCS', (400, 400), cv2.FONT_HERSHEY_PLAIN, 20, (248, 248, 246), 2)
# cv2.imwrite('torcs_title.png', title)

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

title = cv2.resize(cv2.imread('torcs_title.png'), (1920, 1080))
video = cv2.VideoWriter('torcs.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (1920, 1080), True)

# for i in range(72):
#     video.write(title)

for i in range(1000):
    frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 25
    obs = cv2.resize(cv2.imread('demo/%d/obs.png' % i), (1370, 1080))
    frame[:, :1370] = obs
    action, speed = read_action('demo/%d/outcome/log.txt' % i)
    frame = draw_action(frame, 780, 1000, 160, 2, action[0])
    for j in range(10):
        seg = cv2.resize(cv2.imread('demo/%d/outcome/seg%d.png' % (i, j+1)), (265, 208))
        seg = draw_action(seg, 150, 190, 25, 1, action[j])
        cv2.putText(seg, 'Step %d' % (j + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 248, 246), 2)
        cv2.putText(seg, 'speed: %0.2f' % (speed[j]), (10, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (248, 248, 246), 2)
        frame[(int(j/2)*218):(int(j/2)*218+208), (int(j%2)*275+1380):(int(j%2)*275+1380+265)] = seg
    video.write(frame)

video.release()