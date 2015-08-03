"""
Plot energies from joint analysis files.

Usage:
    plot_energies.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

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

def plot_energies(energies, t, output_path='./'):
    [KE, w_rms] = energies

    figs = {}
    
    period = 14.151318259413486
    print("period = {}".format(period))
    fig_energies = plt.figure(figsize=(16,8))
    ax1 = fig_energies.add_subplot(2,1,1)
    #ax1.semilogy(t/period, KE, label="KE")
    ax1.plot(t/period, KE, label="KE")

    gamma_w, log_w0 = compute_growth(w_rms, t, period)
   
    ax2 = fig_energies.add_subplot(2,1,2)
    ax2.semilogy(t/period, w_rms, label=r"$w_{rms}$")
    ax2.semilogy(t/period, np.exp(log_w0)*np.exp(gamma_w*t), 'k-.', label='$\gamma_w = %f$' % gamma_w)
    ax2.legend(loc='lower right').draw_frame(False)

    figs["energies"]=fig_energies


    for key in figs.keys():
        figs[key].savefig('./'+'scalar_{}.png'.format(key))
    
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

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    print(output_path)

    files = args['<files>']
    ts = read_timeseries(files)
    plot_energies([ts['KE'], ts['w_rms']], ts['time'], output_path=output_path)


