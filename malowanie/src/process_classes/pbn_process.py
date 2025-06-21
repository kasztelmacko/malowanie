import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import imageio as iio
import torch
from fast_pytorch_kmeans import KMeans
from skimage.morphology import skeletonize
from skimage.filters import threshold_local
from skimage.measure import label
import cv2

class PaintByNumberProcess:
    def __init__(self, scaled_image, blank_canvas, n_clusters, denoising_kernel_size, min_region_size, font_color, font_scale, font_thickness):
        self.scaled_image = scaled_image
        self.blank_canvas = blank_canvas
        self.n_clusters = n_clusters
        self.denoising_kernel_size = denoising_kernel_size
        self.min_region_size = min_region_size
        self.font_color = font_color
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        
    def apply_kmeans(self, image_array: np.array = None):
        if image_array is None:
            image_array = self.scaled_image

        h, w, c = image_array.shape
        image_tensor = torch.from_numpy(image_array).float()
        pixel_values = image_tensor.view(-1, c)

        self.kmeans = KMeans(n_clusters=self.n_clusters, mode='euclidean', verbose=0)
        self.kmeans_labels = self.kmeans.fit_predict(pixel_values)
        self.kmeans_centroids = self.kmeans.centroids

        quantized_pixels = self.kmeans_centroids[self.kmeans_labels]
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
        gradient_np = gradient_magnitude.numpy() if torch.is_tensor(gradient_magnitude) else gradient_magnitude

        adaptive_thresh = threshold_local(gradient_np, block_size=101, method='gaussian')
        edges = (gradient_np > adaptive_thresh).astype(np.uint8)

        outline_image = self.blank_canvas.copy()
        outline_image[skeletonize(edges)] = list(self.font_color)
        
        return outline_image

    def create_color_palette(self):
        if self.kmeans_centroids is None or self.kmeans_labels is None:
            raise ValueError("KMeans must be run before calling create_color_palette.")

        h, w, _ = self.scaled_image.shape
        indexed_image = self.kmeans_labels.view(h, w).numpy()
        palette = self.kmeans_centroids.numpy().astype(np.uint8)

        return palette, indexed_image
    

    def annotate_regions_with_numbers(self, outline_image, indexed_image):
        annotated = outline_image.copy()

        for cluster_id in range(self.n_clusters):
            mask = (indexed_image == cluster_id)
            labeled_regions = label(mask)

            for region_id in range(1, labeled_regions.max() + 1):
                region_mask = labeled_regions == region_id
                coords = np.argwhere(region_mask)
                if coords.shape[0] < self.min_region_size:
                    continue
                
                centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                cv2.putText(
                    annotated,
                    str(cluster_id + 1),
                    (centroid_x, centroid_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=self.font_scale,
                    color=self.font_color,
                    thickness=self.font_thickness,
                    lineType=cv2.LINE_AA
                )

        return annotated
        
    def generate(self):
        kmeans_image = self.apply_kmeans(image_array=self.scaled_image)
        denoised_image = self.remove_noise_artifacts(image_array=kmeans_image, kernel_size=self.denoising_kernel_size)
        outline_image = self.create_image_outline(image_array=denoised_image)
        palette, indexed_image = self.create_color_palette()
        numbered_outline = self.annotate_regions_with_numbers(outline_image, indexed_image)

        return denoised_image, numbered_outline, palette