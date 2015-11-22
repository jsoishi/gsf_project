"""
Plot spacetime diagrams

Usage:
    plot_spacetime.py <files>... [--output=<dir>  --omega1=<omega1>]

Options:
    --output=<dir>  Output directory 
    --omega1=<omega1> inner rotation freqency   [default: 0.010101010101010102]
"""
import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_energies import read_timeseries

def plot_spacetime(t, x, w):
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    img = ax.pcolormesh(t, x, w.T, cmap='viridis')

    cb = fig.colorbar(img, pad=0.005)
    cb_ax = cb.ax
    ax.set_xlim(0,t[-1])
    ax.set_xlabel(r'$t$',fontsize=24)
    ax.set_ylabel(r'$r$',rotation='horizontal',fontsize=24,labelpad=20)
    cb_ax.set_ylabel(r'$w$',fontsize=24,rotation='horizontal',labelpad=20)

    return fig


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    p = pathlib.Path(args['<files>'][0])
    basename = p.parts[-3]
    print(basename)
    if not args['--output']:
        output_path = pathlib.Path('scratch',p.parts[-3])
        output_path = pathlib.Path(output_path)
    else:
        output_path = pathlib.Path(args['--output']).absolute()

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    print(output_path)

    omega1 = float(args['--omega1'])
    period = 2*np.pi/omega1

    files = args['<files>']
    ts = read_timeseries(files)
    
    with h5py.File(files[-1],'r') as f:
        r = f['scales']['r']['1.0'][:]
    
    fig = plot_spacetime(ts['time']/period, r, ts['w_rms'][:,0,:])

    outfile = output_path.joinpath('{}_w_spacetime.png'.format(basename))
    fig.savefig(str(outfile))
