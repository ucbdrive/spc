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
    action = np.zeros(12)
    action[0] = int(s[0][13])
    for j in range(1, 12):
        action[j] = int(s[2 + 4 * j][7])
    reward = np.zeros(12)
    for j in range(12):
        reward[j] = float(s[3 + 4*j][8:-1])
    value = np.zeros(12)
    for j in range(12):
        value[j] = float(s[4+4*j][7:-1])
    return action, reward, value

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

video = cv2.VideoWriter('flappybird.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (1920, 1080), True)

# for i in range(72):
#     video.write(title)

for i in range(1000):
    frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
    obs = cv2.resize(cv2.imread('demo/%d/obs.png' % i), (820, 1080))
    frame[:, :820] = obs
    action, reward, value = read_action('demo/%d/outcome/log.txt' % i)
    for j in range(12):
        seg = cv2.resize(cv2.imread('demo/%d/outcome/seg%d.png' % (i, j+1)), (265, 354))
        # seg = draw_action(seg, 150, 190, 25, 1, action[j])
        cv2.putText(seg, 'Step %d' % (j + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 248, 246), 2)
        cv2.putText(seg, 'Action:', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (248, 248, 246), 2)
        if action[j] > 0:
            cv2.arrowedLine(seg, (110, 65), (110, 40), (0, 0, 255), 3, tipLength=0.5)
        else:
            cv2.putText(seg, '-', (95, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (248, 248, 246), 2)
        # cv2.putText(seg, 'reward: %0.2f' % (reward[j]), (10, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(seg, 'value: %0.2f' % (value[j]), (10, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        frame[(int(j/4)*363):(int(j/4)*363+354), (int(j%4)*275+830):(int(j%4)*275+830+265)] = seg
    video.write(frame)

video.release()