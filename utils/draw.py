import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


def generate_guide(patch, distribution, radius, line_width=2):
    height, width, _ = patch.shape
    center = (int(width / 2), height)
    for i in range(5):
        r = int(radius / 5 * (5-i))
        cv2.ellipse(patch, center, (r+line_width, r+line_width), 0, -180, 0, (255, 255, 255), -1)
        for j in range(5):
            cv2.ellipse(patch, center, (r, r), 0, -180+36*j, -180+36*j+36, (0, distribution[j*5+4-i]*255, 0), -1)
    for i in range(4):
        angle = math.pi / 5 * (4-i)
        endpoint = (int(center[0] + math.cos(angle) * radius), int(center[1] - math.sin(angle) * radius))
        cv2.line(patch, endpoint, center, (255, 255, 255), line_width)
    return patch



def draw_guide(img, distribution=np.arange(25)/25, radius=320, line_width=4):
    # img = cv2.imread(fname).astype(np.float64)
    height, width, _ = img.shape
    center = (int(width / 2), height)
    patch = img[height-radius-line_width:height, int(width/2-radius-line_width):int(width/2+radius+line_width), :].copy()
    patch = generate_guide(patch, distribution, radius, line_width)
    img[height-radius-line_width:height, int(width/2-radius-line_width):int(width/2+radius+line_width), :] = img[height-radius-line_width:height, int(width/2-radius-line_width):int(width/2+radius+line_width), :] * 0.5 + patch * 0.5
    return img


if __name__ == '__main__':
    draw_guide()