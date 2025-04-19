import numpy as np

class CanvasBase:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.canvas = self.create_blank_canvas()
        
    def create_blank_canvas(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), 255, dtype=np.uint8)