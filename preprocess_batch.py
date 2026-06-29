import os, numpy as np
from skimage.io import imread
from skimage.transform import resize

test_path = 'C:/Users/teoan/Desktop/Practica/arsuri_database/simil/color-full'
SIZE = 64

test_files = sorted([f for f in os.listdir(test_path)
                     if f.startswith('RGB_') and f.endswith('.bmp')
                     and os.path.isfile(os.path.join(test_path, f))])

batch = np.zeros((len(test_files), 3, SIZE, SIZE), dtype=np.float32)
for i, fname in enumerate(test_files):
    img = imread(os.path.join(test_path, fname))[:, :, :3]
    img = resize(img, (SIZE, SIZE), mode='constant', preserve_range=True)
    batch[i] = np.transpose(img, (2, 0, 1)) / 255.0      # CHW, [0,1] 

np.save("fpga_test_inputs.npy", batch)
np.save("fpga_test_names.npy", np.array(test_files))    
print("Salvat:", batch.shape, batch.dtype,
      "range", round(float(batch.min()), 4), round(float(batch.max()), 4))