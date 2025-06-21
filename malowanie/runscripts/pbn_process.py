#%%
from malowanie.src.pbn_process.pbn_process import CreatePaintByNumberProcess
import malowanie.config as PBN_CONF
import traceback
#%%
if __name__ == "__main__":
    try:
        pbn_process = CreatePaintByNumberProcess()
        pbn_process.generate(
            target_image_name="test.jpg",
            scaled_max_width=PBN_CONF.SCALED_MAX_WIDTH,
            n_colors=PBN_CONF.N_COLORS,
            denoising_kernel_size=PBN_CONF.DENOSING_KERNEL_SIZE,
            min_region_size=PBN_CONF.MIN_REGION_SIZE,
            font_color=PBN_CONF.FONT_COLOR,
            font_scale=PBN_CONF.FONT_SCALE,
            font_thickness=PBN_CONF.FONT_THICKNESS
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    
# %%
