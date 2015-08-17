"""
Plot energies from joint analysis files.

Usage:
    plot_energies.py <files>... [--output=<dir> --omega1=<omega1>]

Options:
    --output=<dir>    Output directory
    --omega1=<omega1> inner rotation freqency   [default: 0.010101010101010102]
"""
import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
def read_timeseries(files, verbose=False):
    """Read one-dimensional (f(t)) time series data from Dedalus outputfiles.

    """
    data_files = sorted(files, key=lambda x: int(os.path.split(x)[1].split('.')[0].split('_s')[1]))
    if verbose:
        f = h5py.File(data_files[0], flag='r')
        print(10*'-'+' tasks '+10*'-')
        for task in f['tasks']:
            print(task)
        print(10*'-'+' scales '+10*'-')
        for key in f['scales']:
            print(key)

    f = h5py.File(data_files[0],flag='r')
    ts = {}
    for key in f['tasks'].keys():
        ts[key] = np.array([])

    ts['time'] = np.array([])
    f.close()

    for filename in data_files:
        f = h5py.File(filename, flag='r')
        for k in f['tasks'].keys():
            ts[k] = np.append(ts[k], f['tasks'][k][:])

        ts['time'] = np.append(ts['time'],f['scales']['sim_time'][:])
        f.close()

    return ts

def plot_energies(energies, t, period,output_path='./', calc_growth_rate=False, growth_start=1, growth_stop=2):
    [KE, u_rms, w_rms] = energies

    figs = {}
    
    print("period = {}".format(period))
    fig_energies = plt.figure(figsize=(16,8))
    ax = fig_energies.add_axes([0.1,0.1,0.8,0.8])
    ax.semilogy(t/period, w_rms, label=r"$w_{rms}$")
    ax.semilogy(t/period, u_rms, label=r"$u_{rms}$")
    ax.set_ylabel(r"$< w >_{rms}$", fontsize=20)
    ax.set_xlabel(r"$t/t_{1}$", fontsize=20)
    ax.set_ylim(1e-6,1e-2)
    if calc_growth_rate:
        gamma_w, w0 = compute_growth(w_rms, t, period, growth_start, growth_stop)
        ax.semilogy(t/period, w0*np.exp(gamma_w*t), 'k-.', label='$\gamma_w/\Omega_1 = %f$' % (gamma_w*period/(2*np.pi)))
        ax.legend(loc='lower right').draw_frame(False)

    figs["energies"]=fig_energies


    for key in figs.keys():
        outfile = str(output_path.joinpath('scalar_{}.png'.format(key)))
        figs[key].savefig(outfile)
    
def compute_growth(f, t, period, start, stop, g_scale=80., verbose=True):
    """compute a growth rate gamma for given timeseries f sampled at
    points t, assuming an exponential growth:
    
    f(t) = f0 exp(gamma t)

    inputs:
    f -- timeseries
    t -- time points
    period -- the unit for t
    start -- beginning of timeseries to fit in units of period
    stop -- end of timeseries to fit in units of period

    outputs:
    f0 -- t=0 value
    gamma -- growth rate

    """
    t_window = (t/period > start) & (t/period < stop)

    gamma_f, log_f0 = np.polyfit(t[t_window], np.log(f[t_window]),1)

    return gamma_f, np.exp(log_f0)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    if not args['--output']:
        print('hello?')
        p = pathlib.Path(args['<files>'][0])
        print(p)
        output_path = pathlib.Path('scratch',pathlib.Path(args['<files>'][0]).parts[-3])
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
    plot_energies([ts['KE'], ts['u_rms'], ts['w_rms']], ts['time'], period, output_path=output_path,calc_growth_rate=True)


