from malowanie.src.process_classes.process_base import ProcessBase, InitializePainterProcess
from malowanie.src.process_classes.image_base import ImageBase

class CreatePaintingFromBlankCanvasProcess(ProcessBase):
    def generate(self, target_image_name: str):
        image = ImageBase(image_name = target_image_name)
        process_initializator = InitializePainterProcess(
            target_image = image,
            scaled_max_width = 1024
            )
        scaled_image, blank_cavas = process_initializator.generate()
        image.show_image(scaled_image)
        image.show_image(blank_cavas)