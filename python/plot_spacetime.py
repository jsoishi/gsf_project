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

def read_timeseries(files, verbose=False):
     """Read one-dimensional (f(t)) time series data from Dedalus outputfiles.

     """
     data_files = sorted(files, key=lambda x: int(os.path.split(x)[1].split('.')[0].split('_s')[1]))
     f = h5py.File(data_files[0],'r')
     if verbose:
         print(10*'-'+' tasks '+10*'-')
         for task in f['tasks']:
             print(task)
         print(10*'-'+' scales '+10*'-')
         for key in f['scales']:
             print(key)

     ts = {}
     for key in f['tasks'].keys():
         ts[key] = np.squeeze(f['tasks'][key][:])
     ts['time'] = f['scales']['sim_time'][:]
     f.close()

     for filename in data_files[1:]:
         f = h5py.File(filename)
         for k in f['tasks'].keys():
             ts[k] = np.append(ts[k], np.squeeze(f['tasks'][k][:]),axis=0)

         ts['time'] = np.append(ts['time'],f['scales']['sim_time'][:])
         f.close()

     return ts

def plot_spacetime(t, x, w):
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    img = ax.pcolormesh(t, x, w.T, cmap='viridis')

    cb = fig.colorbar(img, pad=0.005)
    cb_ax = cb.ax
    ax.set_xlim(0,t[-1])
    ax.set_xlabel(r'$t$',fontsize=24)
    ax.set_ylabel(r'$r$',rotation='horizontal',fontsize=24,labelpad=20)
    cb_ax.set_ylabel(r'$<w>$',fontsize=24,rotation='horizontal',labelpad=20)

    return fig

def plot_ztavg(t, x, w,start=0.):
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    wavg = (w[(t > start)]).mean(axis=0)

    
    ax.plot(x,wavg)
    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'$r$',fontsize=24)
    ax.set_ylabel(r'$<w>$',rotation='horizontal',fontsize=24,labelpad=20)

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
        p = pathlib.Path(args['<files>'][0])
        print(p)
        AnalysisDir = pathlib.Path(p.parents[1]/"Analysis")
        if AnalysisDir.is_dir() == False:
            pathlib.Path(AnalysisDir).mkdir()
    output_path = AnalysisDir

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
    
    fig = plot_spacetime(ts['time']/period, r, ts['w_rms'][:,:])
    outfile = output_path.joinpath('{}_w_spacetime.png'.format(basename))
    fig.savefig(str(outfile))

    fig = plot_ztavg(ts['time']/period, r, ts['w_rms'][:,:],start=8.)
    outfile = output_path.joinpath('{}_w_ztavg.png'.format(basename))
    fig.savefig(str(outfile))
