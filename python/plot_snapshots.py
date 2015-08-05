"""
Plot snapshots 

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory 

"""
import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_energies import read_timeseries

def create_frame(r, z, u, v, w):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    img = ax.pcolormesh(r, z,v,cmap='PuOr')
    img.axes.axis('image')
    ax.quiver(r, z, u, w, width=0.005)
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

    # by default, try to stick a "frames" directory in what we guess
    # is the parent directory of slices
    if not args['--output']:
        p = pathlib.Path(args['<files>'][0])
        print(p)
        basepath = pathlib.Path(args['<files>'][0]).parts[-3]
        basepath = pathlib.Path(basepath)
    else:
        basepath = pathlib.Path(args['--output']).absolute()
    output_path = basepath / 'frames'
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    print(output_path)

    files = args['<files>']
    data_files = sorted(files, key=lambda x: int(os.path.split(x)[1].split('.')[0].split('_s')[1]))
    #ts = read_timeseries(files)

    count = 0
    for f in data_files:
        ts = h5py.File(f,'r')

        r = ts['scales']['r']['1.0'][:]
        z = ts['scales']['z']['1.0'][:]

        for i in range(ts['tasks']['v'][:].shape[0]):
            u = ts['tasks']['u'][i,:]
            v = ts['tasks']['v'][i,:]
            w = ts['tasks']['w'][i,:]
            fig = create_frame(r,z,u,v,w)
            print("saving frame {:04d}".format(count))
            filen = str(output_path / "vel_frame_{:04d}".format(count))
            fig.savefig(filen)
            plt.close()
            count += 1
        ts.close()




