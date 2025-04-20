#%%
from malowanie.src.pbn_process.pbn_process import CreatePaintByNumberProcess
import malowanie.config as CONF
import traceback
#%%
if __name__ == "__main__":
    try:
        pbn_process = CreatePaintByNumberProcess()
        pbn_process.generate(
            target_image_name="Lenna.png",
            scaled_max_width=CONF.SCALED_MAX_WIDTH,
            n_colors=CONF.N_COLORS,
            denoising_kernel_size=CONF.DENOSING_KERNEL_SIZE,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    
# %%
