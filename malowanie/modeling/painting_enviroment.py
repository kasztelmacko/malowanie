from malowanie.modeling.painting_base.canvas_base import CanvasBase
from malowanie.modeling.painting_base.image_base import ImageBase

class PaintingEnviroment:
    def __init__(self, image_path: str):
        self.target_image  = ImageBase(
            image_path=image_path
        )
        self.blank_canvas = CanvasBase(
            height=self.target_image.height,
            width=self.target_image.width,
        )