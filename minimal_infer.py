import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

# 1) Inject CLI args so that parse() sees --dataroot etc.
sys.argv = [
    sys.argv[0],
    '--dataroot', './datasets2/gelsigh_mini_nocs_20_contact',
    '--name', 'tactile_mnisq_init_cyclegan8_norm_noprocess',
    '--model', 'test',
    '--no_dropout',
    '--num_test', '1',
    '--checkpoints_dir', './checkpoints'
]

# 2) Parse options (no args parameter—parse() reads sys.argv) 
from options.test_options import TestOptions
opt = TestOptions().parse()  # uses the flags we just injected :contentReference[oaicite:0]{index=0}

# 3) Build and prepare the model
from models import create_model
model = create_model(opt)
model.setup(opt)
model.eval()

# 4) Load & preprocess your single input image
img = Image.open('./datasets2/gelsigh_mini_nocs_20_contact/AtestA_NOCS/04592_000_12.jpeg').convert('RGB')

transform = transforms.Compose([
    transforms.Resize(opt.load_size),                   # same as test.py
    transforms.ToTensor(),                              # [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # →[-1,1]
])
input_tensor = transform(img).unsqueeze(0)  # [1,3,H,W]

# 5) Run inference
with torch.no_grad():
    fake_tensor = model.netG(input_tensor)  # A→B direction

# 6) Denormalize & save
fake_tensor = (fake_tensor + 1.0) / 2.0        # back to [0,1]
to_pil = transforms.ToPILImage()
fake_img = to_pil(fake_tensor.squeeze(0))
fake_img.save('output.jpg')
print('✅ Inference complete — saved to output.jpg')

