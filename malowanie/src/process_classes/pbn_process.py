import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import imageio as iio
import torch
from fast_pytorch_kmeans import KMeans

class PaintByNumberProcess:
    def __init__(self, scaled_image, blank_canvas, n_clusters, denoising_kernel_size):
        self.scaled_image = scaled_image
        self.blank_canvas = blank_canvas
        self.n_clusters = n_clusters
        self.denoising_kernel_size = denoising_kernel_size
        
    def apply_kmeans(self, image_array: np.array = None):
        """Apply K-means clustering to the input image for color quantization.
        """
        if image_array is None:
            image_array = self.scaled_image
        
        h, w, c = image_array.shape
        image_tensor = torch.from_numpy(image_array).float()
        pixel_values = image_tensor.view(-1, c)

        kmeans = KMeans(n_clusters=self.n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(pixel_values)
        centers = kmeans.centroids
        
        quantized_pixels = centers[labels]
        quantized_image = quantized_pixels.view(h, w, c)
        
        return quantized_image.numpy().astype(np.uint8)
    
    def remove_noise_artifacts(self, image_array: np.array = None, kernel_size: int = 3):
        if image_array is None:
            image_array = self.scaled_image
        
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        
        pad = kernel_size // 2
        padded = np.pad(image_array, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        windows = sliding_window_view(padded, (kernel_size, kernel_size, 1), axis=(0, 1, 2))
        denoised = np.median(windows, axis=(3, 4)).squeeze()
        
        return denoised.astype(np.uint8)
    
    def create_image_outline(self, image_array: np.array = None):
        if image_array is None:
            image_array = self.scaled_image
        
        image_tensor = torch.from_numpy(image_array).float()
        gray = (image_tensor[..., :3] @ torch.tensor([0.2989, 0.5870, 0.1140])).unsqueeze(0)
        
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        
        grad_x = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_y, padding=1)
        gradient_magnitude = (grad_x**2 + grad_y**2).sqrt().squeeze()

        edges = (gradient_magnitude > 30).numpy()

        outline_image = self.blank_canvas.copy()
        outline_image[edges] = [0, 0, 0]
        
        return outline_image
    
    def create_color_palette(self, image_array: np.array = None):
        pass
        

    def generate(self):
        kmeans_image = self.apply_kmeans(image_array=self.scaled_image)
        denoised_image = self.remove_noise_artifacts(image_array=kmeans_image, kernel_size=self.denoising_kernel_size)
        outline_image = self.create_image_outline(image_array=denoised_image)
        return denoised_image, outline_image