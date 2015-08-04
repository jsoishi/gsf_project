"""
Plot snapshots 

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""
import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_energies import read_timeseries

def create_frame(r, z, u, v, w):
    fig = plt.figure(figsize=(8,12))
    ax = fig.add_subplot(111)
    img = ax.pcolormesh(r, z,v,cmap='PuOr')
    img.axes.axis('image')
    ax.quiver(r, z, u, v, width=0.005)
    cb = fig.colorbar(img, pad=0.005)
    cb_ax = cb.ax
    for a in [cb_ax, ax]:
        a.tick_params(labelsize=12)

    for axis in ['left','right']:
        ax.spines[axis].set_linewidth(4.5)

    ax.set_xlabel(r'$r$',fontsize=24)
    ax.set_ylabel(r'$z$',rotation='horizontal',fontsize=24,labelpad=20)
    cb_ax.set_ylabel(r'$v_{\theta}$',fontsize=24,rotation='horizontal',labelpad=20)

    return fig


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    output_path = output_path / 'frames'
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    print(output_path)

    files = args['<files>']
    #ts = read_timeseries(files)
    ts = h5py.File(files[0],'r')

    r = ts['scales']['r']['1.0'][:]
    z = ts['scales']['z']['1.0'][:]



    for i in range(ts['tasks']['v'][:].shape[0]):
        u = ts['tasks']['u'][i,:]
        v = ts['tasks']['v'][i,:]
        w = ts['tasks']['w'][i,:]
        fig = create_frame(r,z,u,v,w)

        filen = str(output_path / "vel_frame_{:04d}".format(i))
        fig.savefig(filen)




