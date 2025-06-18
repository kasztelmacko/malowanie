from abc import ABC, abstractmethod
from malowanie.src.process_classes.image_base import ImageBase

class ProcessBase(ABC):
    @abstractmethod
    def generate(self):
        pass
    
class InitializePainterProcess(ProcessBase):
    def __init__(self, target_image: ImageBase, scaled_max_width: int):
        self.target_image = target_image
        self.scaled_max_width = scaled_max_width

    def generate(self):
        original_width = self.target_image.width
        original_height = self.target_image.height

        if original_width > self.scaled_max_width:
            scaling_factor = self.scaled_max_width / original_width
            new_height = int(original_height * scaling_factor)
            new_width = self.scaled_max_width
        else:
            new_width, new_height = original_width, original_height

        scaled_image = self.target_image.resize(new_width, new_height)
        blank_canvas = self.target_image.create_blank_canvas(new_width, new_height)
        return scaled_image, blank_canvas