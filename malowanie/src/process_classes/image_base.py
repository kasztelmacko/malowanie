import imageio as iio
from malowanie import config as CONF
import numpy as np
from skimage.transform import resize as skimage_resize
import matplotlib.pyplot as plt

class ImageBase:
    def __init__(self, image_name: str):
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