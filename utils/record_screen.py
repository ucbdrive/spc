import numpy as np
import time
import cv2
import mss


def shot(height, width):
    with mss.mss() as sct:
        img = np.array(
            sct.grab(
                {'top': 0,
                 'left': 0,
                 'width': width,
                 'height': height}
                )
            )[:, :, :3]
    return img


def record_screen(signal, fname, width, height, frame_rate=24.0):
    video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (width, height), True)
    while signal.value == 1:
        video.write(shot(height, width))
    video.release()
