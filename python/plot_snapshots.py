"""
Plot snapshots 

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory 

notes: use 

ffmpeg -framerate 20 -i write_%04d.png -s:v 1600x1200 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p gsf.mp4


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
def create_frame(r, z, u, v, w, T, time, quiver=False):
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_axes([0.1,0.1,0.35,0.8])
    img = ax.pcolormesh(r, z,v,cmap='YlGnBu')
    img.axes.axis('image')
    ax.set_title('t = {:10.5e}'.format(time),fontsize=24)

    skip = (slice(None, None, 3), slice(None, None, 3))
    if quiver:
        ax.quiver(r[skip[0]], z[skip[0]], u[skip], w[skip], width=0.005)
    cb = fig.colorbar(img, pad=0.005)
    cb_ax = cb.ax

    ax2 = fig.add_axes([0.55,0.1,0.35,0.8])
    img2 = ax2.pcolormesh(r,z,T,cmap='bwr')
    img2.axes.axis('image')
    cb2 = fig.colorbar(img2, pad=0.005)
    cb2_ax = cb2.ax
    for a in [cb_ax, ax, ax2, cb2_ax]:
        a.tick_params(labelsize=16)

    for a in [ax,ax2]:
        for axis in ['left','right']:
            a.spines[axis].set_linewidth(4.5)

        a.set_xlabel(r'$r$',fontsize=24)
        a.set_ylabel(r'$z$',rotation='horizontal',fontsize=24,labelpad=20)
    cb_ax.set_ylabel(r'$v_{\theta}$',fontsize=24,rotation='horizontal',labelpad=20)
    cb2_ax.set_ylabel(r'$T_1$', fontsize=24,rotation='horizontal',labelpad=20)
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
        AnalysisDir = pathlib.Path(p.parents[1]/"Analysis")
        if AnalysisDir.is_dir() == False:
            pathlib.Path(AnalysisDir).mkdir()
        if pathlib.Path(AnalysisDir/"frames").is_dir() == False:
            pathlib.Path(AnalysisDir/"frames").mkdir()
        #basepath = pathlib.Path('scratch',pathlib.Path(args['<files>'][0]).parts[-3])
       # basepath = pathlib.Path(basepath)
    #else:
       # basepath = pathlib.Path(args['--output']).absolute()
    output_path = AnalysisDir / 'frames'
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
        time = ts['scales']['sim_time']

        for i in range(ts['tasks']['v'][:].shape[0]):
            u = ts['tasks']['u'][i,:]
            v = ts['tasks']['v'][i,:]
            w = ts['tasks']['w'][i,:]
            T = ts['tasks']['T'][i,:]
            tt = time[i]
            fig = create_frame(r,z,u,v,w, T, tt)
            print("saving frame {:04d}".format(count))
            filen = str(output_path / "vel_frame_{:04d}".format(count))
            fig.savefig(filen)
            plt.close()
            count += 1
        ts.close()




