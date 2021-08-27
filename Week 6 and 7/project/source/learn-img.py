from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")


image_name = "7ba08d63-79fe-416d-b435-f3d3fdec5f3c.bmp"

image_path_full = os.path.join(train_src_image_folder, image_name)
print(f"src = {image_path_full}")
label_path_full = os.path.join(train_label_image_folder, image_name)
print(f"label = {label_path_full}")

src_image = Image.open(image_path_full)
print(src_image.size)
src_image_rgb = src_image.convert("RGB")
#src_image_rgb.show()

label_image = Image.open(label_path_full)
print(label_image.size)
#label_image.show()
src_image_data = np.array(src_image, dtype = np.float32)
src_image_data = src_image_data / 255

label_image_data = np.array(label_image, dtype = np.float32)
print(label_image_data.max())
label_image_data[label_image_data>0]=1.0
print(label_image_data.max())

figure, sub_image_arr = plt.subplots(2)
sub_image_arr[0].imshow(src_image_data, cmap = "gray")
sub_image_arr[1].imshow(label_image_data, cmap = "gray")
plt.show()

image_width = 716//4
image_height = 920//4
src_image = Image.open(image_path_full).convert("RGB")
preprocess = transforms.Compose([
                                    transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229, 0.224, 0.229])])

src_image_data=preprocess(src_image)
print(src_image_data.shape)
print(src_image_data)


label_image = Image.open(label_path_full).convert("L")

label_image_resized = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)(label_image)

label_image_data = np.array(label_image_resized)
print(label_image_data)
print(label_image_data.max())

defect_class = np.zeros_like(label_image_data)
defect_class[label_image_data>0]=1.0

background_class = np.zeros_like(label_image_data)
background_class[label_image_data==0]=1.0

label_image_classes = torch.tensor([background_class, defect_class])
print(label_image_classes.shape)
