import matplotlib.pyplot as plt
import h5py 
import sys
from pathlib import Path
filename = Path(sys.argv[-1])
df = h5py.File(filename,"r")
t = df["scales/sim_time"][:]
KE_fluct = df["tasks/KE_fluct"][:].ravel()
KE_fluct_zonal_v = df["tasks/KE_fluct_zonal_v"][:].ravel()
KE_fluct_zonal_w = df["tasks/KE_fluct_zonal_w"][:].ravel()

plt.semilogy(t, KE_fluct, label="Total KE")
plt.semilogy(t,KE_fluct_zonal_v, label = "Zonal KE v")
plt.semilogy(t,KE_fluct_zonal_w, label = "Zonal KE w")
plt.legend()
outpath = Path(filename.parents[1]/"Analysis")
if outpath.is_dir() == False:
    Path(outpath).mkdir()
plt.savefig(outpath/"KEfig.png")
        
