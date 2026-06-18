
import os
import platform
import psutil
from unittest import mock
import os
import psutil
import tracemalloc
import time
import numpy as np
from astropy.io import fits

# ---- Simulated system specs ----
SIM_CORES = 2
SIM_MEMORY_GB = 8  # choose a realistic number for your fake system
SIM_CPU = "Intel(R) Celeron(R) 4305UE @ 2.00GHz"

# ---- Helper functions to "simulate" ----
def get_fake_system_info():
    return {
        "System": platform.system(),
        "Node Name": "fake-celeron4305ue",
        "Release": platform.release(),
        "Machine": platform.machine(),
        "Processor": SIM_CPU,
        "Cores": SIM_CORES,
        "Threads": SIM_CORES,  # no hyperthreading
        "Total Memory (GB)": SIM_MEMORY_GB,
        "Max Memory Channels": 2,
    }


# ---- Example usage ----
if __name__ == "__main__":

    fake_info = get_fake_system_info()
    for key, value in fake_info.items():
        print(f"{key}: {value}")
    
    with mock.patch("os.cpu_count", return_value=2), \
        mock.patch("psutil.virtual_memory", return_value=psutil._pslinux.svmem(
            total=8*1024**3, available=8*1024**3, percent=0, used=0, free=8*1024**3,
            active=0, inactive=0, buffers=0, cached=0, shared=0, slab=0)):
        
        print("Simulated CPU count:", os.cpu_count())
        print("Simulated RAM (GB):", psutil.virtual_memory().total / 1024**3)

    tracemalloc.start()
    start = time.time()

    #Your code here !!!

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    end = time.time()
    timing = end - start
    print(f'Run  in {np.round(timing,2)} sec! ')

