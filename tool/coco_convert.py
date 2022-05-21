import os.path
import cv2 as cv
import json
import numpy as np

data_path = r'D:\TextOCR\TextOCR_0.1_val.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
image = {}
for key, value in data['imgs'].items():
    image[key] = {
        "file_name": value['file_name'].split("/")[-1],
        "original_size": [value['width'], value['height']],
        "target": []
    }
for key, value in data['anns'].items():
    image[value['image_id']]['target'].append({
        "bbox": np.array(value['points']).reshape((-1, 2)).tolist(),
        "text": value['utf8_string']
    })
save_image_dir = r"D:\TextOCR\splited\valid\image"
image_dir = r'D:\TextOCR\train_val_images\train_images'
for key, value in image.items():
    tmp = cv.imread(os.path.join(image_dir, value['file_name']))
    cv.imwrite(os.path.join(save_image_dir, value['file_name']), tmp)
target_file = r"D:\TextOCR\splited\valid\target.json"
with open(target_file, "w", encoding='utf-8') as f:
    f.write(json.dumps(list(image.values())))

