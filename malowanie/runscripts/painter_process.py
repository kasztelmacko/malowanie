#%%
from malowanie.src.painter_process.painter_process import CreatePaintingFromBlankCanvasProcess
import traceback

#%%
if __name__ == "__main__":
    try:
        painter_process = CreatePaintingFromBlankCanvasProcess()
        painter_process.generate(
            target_image_name="TEST.jpg"
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
# %%
