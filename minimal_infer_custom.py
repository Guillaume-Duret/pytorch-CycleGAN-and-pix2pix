import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from options.test_options import TestOptions
from models import create_model

def is_image_black(image_path):
    """Check if the image is completely black."""
    with Image.open(image_path) as img:
        np_img = np.array(img)
    return np.all(np_img == 0)

def preprocess_nocs_image(depth_image, isblack, to_resize=True, process_NOCS=True):
    """ma homie, this takes your already-loaded PIL depth_image and spits RGB NOCS back."""
    # get original dims
    orig_w, orig_h = depth_image.size

    # resize if needed
    if to_resize:
        # keep it 240×240 so da net don’t trip
        depth_image = depth_image.resize((240, 240), Image.ANTIALIAS)

    # to numpy single-channel
    arr = np.array(depth_image)
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # just one channel, bruh

    # set up blank RGB canvas
    h, w = (240, 240) if to_resize else (orig_h, orig_w)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if process_NOCS:
        # paint NOCS coords into R/G/B
        for y in range(h):
            for x in range(w):
                z = 255 if isblack else int(arr[y, x])
                rgb[y, x] = [
                    int(y * 255 / h),  # R = y-norm
                    int(x * 255 / w),  # G = x-norm
                    z                  # B = depth or white if black
                ]
    return Image.fromarray(rgb, 'RGB')

if __name__ == '__main__':
    #—you gotta fake the CLI so TestOptions.parse() doesn’t cry—
    sys.argv = [
        sys.argv[0],
        '--dataroot',        './datasets2/gelsigh_mini_nocs_20_contact',
        '--name',            'tactile_mnisq_init_cyclegan8_norm_noprocess',
        '--model',           'test',
        '--no_dropout',
        '--num_test',        '1',
        '--checkpoints_dir', './checkpoints'
    ]

    # parse & prep model
    opt   = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    model.eval()  # drop dropout/batchnorm shenanigans

    # 1) **load** your depth image first, yo  
    input_path   = './datasets2/gelsigh_mini_nocs_20_contact/testA/00000_000_10.jpeg'
    depth_image  = Image.open(input_path).convert('RGB')
    isblack_flag = is_image_black(input_path)

    # 2) **preprocess** it into NOCS‐RGB
    rgb_img = preprocess_nocs_image(
        depth_image,
        isblack_flag,
        to_resize=True,
        process_NOCS=True
    )

    # 3) now standard CycleGAN transforms
    transform = transforms.Compose([
        transforms.Resize(opt.load_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    inp_tensor = transform(rgb_img).unsqueeze(0)  # [1,3,H,W]

    # 4) run inference
    with torch.no_grad():
        fake_tensor = model.netG(inp_tensor)

    # 5) denorm & save
    fake_tensor = (fake_tensor + 1.0) / 2.0
    to_pil      = transforms.ToPILImage()
    fake_image  = to_pil(fake_tensor.squeeze(0))
    fake_image.save('output.jpg')
    print('✅ Inference + NOCS preprocess done – output.jpg saved')

