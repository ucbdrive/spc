from PIL import Image
from torchvision.transforms import functional as tfn
import cv2


class SegmentationTransform:
    def __init__(self, longest_max_size, rgb_mean, rgb_std):
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def __call__(self, img):
        # Scaling
        # scale = self.longest_max_size/float(max(img.shape[0],img.shape[1]))
        # if scale != 1.:
        #     out_size = tuple(int(dim * scale) for dim in img.shape)
        #     img = cv2.resize(img, size=out_size, mode='bilinear', align_corners=True)

        # Convert to torch and normalize
        img = tfn.to_tensor(img)
        img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        img.div_(img.new(self.rgb_std).view(-1, 1, 1))

        return img
