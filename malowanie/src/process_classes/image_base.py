import imageio as iio
from malowanie import config as CONF
import cv2
import numpy as np
from skimage.transform import resize as skimage_resize
import matplotlib.pyplot as plt

class ImageBase:
    def __init__(self, image_name: str):
        self.image_name = image_name
        self.image_path = CONF.RAW_DATA_DIR / image_name
        self.image_array = self.read_image(self.image_path)
        self.height = self.get_image_height(self.image_array)
        self.width = self.get_image_width(self.image_array)
        
    def read_image(self, image_path: str) -> np.ndarray:
        return iio.imread(image_path)
    
    def get_image_height(self, image_array: np.ndarray) -> int:
        return image_array.shape[0]
    
    def get_image_width(self, image_array: np.ndarray) -> int:
        return image_array.shape[1]
    
    def create_blank_canvas(self, width: int, height: int) -> np.ndarray:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    
    def resize(self, new_width: int, new_height: int) -> np.ndarray:
        resized_image = skimage_resize(
            self.image_array,
            (new_height, new_width),
            anti_aliasing=True
        )
        return (resized_image * 255).astype(np.uint8)
    
    def show_image(self, image_array: np.ndarray = None):
        if image_array is None:
            image_array = self.image_array
        plt.imshow(image_array)
        plt.axis('off')
        plt.show()

    @staticmethod
    def save_image(image_array, image_name, save_path):
        save_dir = save_path
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / image_name
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(file_path), image_array)