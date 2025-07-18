from PIL import Image
import numpy as np

mask = Image.open('Dataset/test/masks/image_23.png').convert('L')
arr = np.array(mask)
print(np.unique(arr))