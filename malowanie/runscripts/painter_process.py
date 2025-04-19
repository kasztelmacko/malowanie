#%%
from malowanie.src.painter_process.painter_process import CreatePaintingFromBlankCanvasProcess

#%%
if __name__ == "__main__":
    try:
        painter_process = CreatePaintingFromBlankCanvasProcess("TEST.jpg")
        painter_process.generate()
    except:
        pass