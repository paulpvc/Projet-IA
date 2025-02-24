import PIL
from PIL import Image
import pandas as pd
import sklearn
import os

path = "/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Mer"
img = Image.open("/home/paul_pvc/PycharmProjects/pythonProject/TP_IA_L3/Init/Mer/838s.jpg")
print("Original size:", img.size)

new_size = (img.width * 2, img.height * 2)  # Upscaling 2x
resized_image = img.resize(new_size, Image.LANCZOS)  # Nearest-neighbor upscaling

print("Resized size:", resized_image.size)