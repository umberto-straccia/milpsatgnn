import os
import math

def clean_folder(folder):
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder)
#<

def get_n_procs(n_procs):
    # if n_procs is int use n_procs else fraction of available cores
    avail = os.cpu_count()
    n_procs = n_procs if isinstance(n_procs, int) else math.floor(n_procs * avail)
    if n_procs == 0:
        n_procs = avail
    elif n_procs < 0:
        n_procs = avail + n_procs
    return n_procs
#<