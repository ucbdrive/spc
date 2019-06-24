import os
from shutil import copyfile
import PIL.Image as Image

demo_dir = "./demo"
demo_dst_dir = "demo_video"

if not os.path.exists(demo_dst_dir):
    os.makedirs(demo_dst_dir)
    os.makedirs(os.path.join(demo_dst_dir, "img"))
    os.makedirs(os.path.join(demo_dst_dir, "combined"))

demo_dirs = os.listdir(demo_dir)

seg_col, seg_row = 2,5
seg_size = 100
obs_size = 500

for img_dir in demo_dirs:
    img_path = os.path.join(demo_dir, img_dir, "obs.png")
    dst_path = os.path.join(demo_dst_dir, "img", "{}.png".format(img_dir.zfill(4)))
    copyfile(img_path, dst_path)
    seg_path = os.path.join(demo_dir, img_dir, "outcome")
    seg_overall = Image.new('RGB', (obs_size + seg_col * seg_size, seg_row * seg_size))
    for i in range(10):
        seg_name = "seg{}.png".format(i+1)
        seg_img_name = os.path.join(seg_path, seg_name)
        from_img = Image.open(seg_img_name)
        # from_img = from_img.convert('RGB')
        from_img = from_img.resize((seg_size, seg_size), Image.ANTIALIAS)
        col_index = i % seg_col
        row_index = i % seg_row
        seg_overall.paste(from_img, (obs_size + col_index * seg_size, row_index * seg_size))
    obs_img = Image.open(img_path)
    obs_img = obs_img.resize((obs_size, obs_size), Image.ANTIALIAS)
    seg_overall.paste(obs_img, (0,0))
    seg_overall.save(os.path.join(demo_dst_dir, "combined", "{}.png".format(img_dir.zfill(4))))

img_target_dir = os.path.join(demo_dst_dir, "combined")
combine_video_cmd = "ffmpeg -f image2 -i {}/%04d.png  -vcodec libx264 -r 10 {}/demo_out.mp4".format(img_target_dir, demo_dst_dir)
os.system(combine_video_cmd)
