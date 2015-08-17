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

def plot_energies(energies, t, period,output_path='./', calc_growth_rate=False):
    [KE, u_rms, w_rms] = energies

    figs = {}
    
    print("period = {}".format(period))
    fig_energies = plt.figure(figsize=(16,8))
    ax1 = fig_energies.add_subplot(2,1,1)
    ax2 = fig_energies.add_subplot(2,1,2)
    #ax1.semilogy(t/period, KE, label="KE")
    ax1.plot(t/period, KE, label="KE")
    ax1.set_ylabel(r"$E_{kin}$", fontsize=20)
    ax2.semilogy(t/period, w_rms, label=r"$w_{rms}$")
    ax2.semilogy(t/period, u_rms, label=r"$u_{rms}$")
    ax2.set_ylabel(r"$< w >_{rms}$", fontsize=20)
    ax2.set_xlabel(r"$t/t_{1}$", fontsize=20)
    if calc_growth_rate:
        gamma_w, log_w0 = compute_growth(w_rms, t, period)
        ax2.semilogy(t/period, np.exp(log_w0)*np.exp(gamma_w*t), 'k-.', label='$\gamma_w = %f$' % gamma_w)
        ax2.legend(loc='lower right').draw_frame(False)

    figs["energies"]=fig_energies


    for key in figs.keys():
        outfile = str(output_path.joinpath('scalar_{}.png'.format(key)))
        figs[key].savefig(outfile)
    
def compute_growth(wrms, t, period, g_scale=80., verbose=True):
    t_window = (t/period > 2) & (t/period < 8)

    gamma_w, log_w0 = np.polyfit(t[t_window], np.log(wrms[t_window]),1)

    if verbose:
        gamma_w_scaled = gamma_w*g_scale
        gamma_barenghi = 0.430108693
        rel_error_barenghi = (gamma_barenghi - gamma_w_scaled)/gamma_barenghi
        print("gamma_w_scaled: {:10.5e}".format(gamma_w_scaled))
        print("rel_error: {:10.5e}".format(rel_error_barenghi))

    return gamma_w, log_w0

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
    plot_energies([ts['KE'], ts['u_rms'], ts['w_rms']], ts['time'], period, output_path=output_path)


