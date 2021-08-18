import os
from PIL import Image
import torchvision.transforms.functional as TF

train_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/srcImg"
train_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/label"

src_image_names = os.listdir(train_src_image_folder)
print(src_image_names)
print(len(src_image_names))

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"
src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")

src_image = Image.open(src_image_full_path)
label_image = Image.open(label_image_full_path)

print("src_image")
src_image.show()
print("label_image")
label_image.show()

src_image_tensor = TF.to_tensor(src_image)
label_image_tensor = TF.to_tensor(label_image)

print("src_image")
src_image.show()
print("label_image")
label_image.show()

src_image_tensor = TF.to_tensor(src_image)
label_image_tensor = TF.to_tensor(label_image)

print(src_image_tensor)
print(src_image_tensor.size())

print(src_image_tensor[0])
