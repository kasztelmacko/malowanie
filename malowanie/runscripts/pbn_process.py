#%%
from malowanie.src.pbn_process.pbn_process import CreatePaintByNumberProcess
import malowanie.config as PBN_CONF
import traceback
#%%
if __name__ == "__main__":
    try:
        pbn_process = CreatePaintByNumberProcess()
        pbn_process.generate(
            target_image_name="marti_test.jpg",
            scaled_max_width=PBN_CONF.SCALED_MAX_WIDTH,
            n_colors=PBN_CONF.N_COLORS,
            denoising_kernel_size=PBN_CONF.DENOSING_KERNEL_SIZE,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    
# %%
