import os
import pickle as pkl
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_data = 'data/images/'

model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

image_to_caption = {}
for img in os.listdir(image_data):
    image_path = os.path.join(image_data, img)
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    gen_caption = model.generate({"image": image})
    image_to_caption[img] = gen_caption[0]

with open('data/blip_gen_captions.pkl', 'wb') as f:
    pkl.dump(image_to_caption, f)