from malowanie.src.process_classes.process_base import ProcessBase, InitializePainterProcess
from malowanie.src.process_classes.image_base import ImageBase
from malowanie.src.process_classes.pbn_process import PaintByNumberProcess

class CreatePaintByNumberProcess(ProcessBase):
    def generate(self, target_image_name: str, scaled_max_width: int, n_colors: int, denoising_kernel_size: int):
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
        denoised_image, outline_image = pbn_process.generate()
        image.show_image(denoised_image)
        image.show_image(outline_image)