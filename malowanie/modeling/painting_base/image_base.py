import numpy as np
import imageio as iio

class ImageBase:
    def __init__(self, image_path: str):
        self.image = self.load_image(image_path)
        self.height, self.width = self.image.shape[:2]
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a file path.
        
        Args:
            image_path (str): The path to the image file.
        
        Returns:
            np.ndarray: The loaded image as a NumPy array.
        """
        img = iio.imread(image_path)
        return img.astype(np.uint8)
    
