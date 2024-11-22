import onnxruntime as ort
import numpy as np
import cv2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

def prepare_input(rgb_image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray], List[int]]:
    h, w = rgb_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    onnx_input = {
        "pixel_values": np.ascontiguousarray(
            np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32
        )
    }
    return onnx_input, pad_info

def infer_single_image(image_path: str, model_path: str, input_size=(616, 1064)):
    session = ort.InferenceSession(model_path)
    
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    onnx_input, pad_info = prepare_input(rgb_image, input_size)
    outputs = session.run(None, onnx_input)
    
    depth = outputs[0].squeeze()
    depth = depth[
        pad_info[0] : input_size[0] - pad_info[1],
        pad_info[2] : input_size[1] - pad_info[3],
    ]
    depth = cv2.resize(depth, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return depth, rgb_image

if __name__ == "__main__":
    model_path = "./weight/metric3d_vit_large.onnx"
    image_path = "./input/147/1/original/0001.jpg"
    
    depth_pred, original_image = infer_single_image(image_path, model_path)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(original_image)
    plt.title('Imagen Original')
    
    plt.subplot(122)
    plt.imshow(depth_pred, cmap='plasma')
    plt.title('Predicción de Profundidad')
    plt.colorbar()
    
    plt.show()