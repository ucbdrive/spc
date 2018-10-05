import numpy as np
import cv2

videos = [[1, 13, 14, 5],
          [9, 11, 16, 6]]
batch = 0

caps = [cv2.VideoCapture('gta_clip_%d.avi' % i) for i in videos[batch]]
video = cv2.VideoWriter('clips%d.avi' % batch, cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (1920, 1080), True)

duration = 10 * 24

height = 535
width = 955

for i in range(duration):
    _frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 25
    ret, frame = caps[0].read()
    _frame[:height, :width, :] = cv2.resize(frame[:, 4:], (width, height))
    ret, frame = caps[1].read()
    _frame[:height, -width:, :] = cv2.resize(frame[:, 4:], (width, height))
    ret, frame = caps[2].read()
    _frame[-height:, :width, :] = cv2.resize(frame[:, 4:], (width, height))
    ret, frame = caps[3].read()
    _frame[-height:, -width:, :] = cv2.resize(frame[:, 4:], (width, height))
    video.write(_frame)

for cap in caps:
    cap.release()
video.release()