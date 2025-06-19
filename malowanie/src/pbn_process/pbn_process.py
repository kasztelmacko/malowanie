from malowanie.src.process_classes.process_base import InitializePainterProcess
from malowanie.src.process_classes.image_base import ImageBase
from malowanie.src.process_classes.pbn_process import PaintByNumberProcess

class CreatePaintByNumberProcess:
    def generate(self, target_image_name=None, scaled_max_width=None, n_colors=None, denoising_kernel_size=None):
        image = ImageBase(image_name=target_image_name)
        process_initializator = InitializePainterProcess(
            target_image=image,
            scaled_max_width=scaled_max_width,
        )
        scaled_image, blank_canvas = process_initializator.generate()
        pbn_process = PaintByNumberProcess(
            scaled_image=scaled_image,
            blank_canvas=blank_canvas,
            n_clusters=n_colors,
            denoising_kernel_size=denoising_kernel_size,
        )
        denoised_image, numbered_outline, palette = pbn_process.generate()
        image.show_image(denoised_image)
        image.show_image(numbered_outline)
        print(palette)