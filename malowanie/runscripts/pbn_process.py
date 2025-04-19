#%%
from malowanie.src.pbn_process.pbn_process import CreatePaintByNumberProcess
#%%
if __name__ == "__main__":
    try:
        pbn_process = CreatePaintByNumberProcess("TEST.jpg")
        pbn_process.generate()
    except:
        pass
    