"""
Plot energies from joint analysis files.

Usage:
    plot_energies.py <files>... [--output=<dir> --eta=<eta> --calc-growth --growth-start=<start> --growth-stop=<stop> --avg-start=<avg_start> --avg-stop=<avg_stop>]

Options:
    --output=<dir>              Output directory
    --eta=<eta>                 eta = R1/R2  [default: 0.99]
    --calc-growth               calculate growth rate
    --growth-start=<start>      start time for growth rate calculation in units of inner cylinder period [default: 0.25]
    --growth-stop=<stop>        stop time for growth rate calculation in units of inner cylinder period [default: 0.5]
    --avg-start=<avg_start>     start time for averaging in units of inner cylinder period [default: 14]
    --avg-stop=<ag_stop>        stop time for averaging in units of inner cylinder period [default: 15]
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

def plot_energy(energy, t, period, basename, avg_start, avg_stop, output_path='./'):
    fig_en = plt.figure(figsize=(16,8))
    ax = fig_en.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(t/period, energy)
    ax.set_xlabel(r"$t/t_{1}$", fontsize=20)
    ax.set_ylabel("Energy", fontsize=20)
    print("average energy from t = {}  to t = {}: {}".format(avg_start, avg_stop, energy[(t/period > avg_start) & (t/period < avg_stop)].mean()))
    outfile = str(output_path.joinpath('{}_energy.png'.format(basename)))
    fig_en.savefig(outfile)

def plot_enstrophy(enstrophy, t, period, basename, avg_start, avg_stop, output_path='./'):
    fig_en = plt.figure(figsize=(16,8))
    ax = fig_en.add_axes([0.1,0.1,0.8,0.8])
    ax.semilogy(t/period, enstrophy)
    ax.set_xlabel(r"$t/t_{1}$", fontsize=20)
    ax.set_ylabel("Enstrophy", fontsize=20)
    print("average enstrophy from t = {}  to t = {}: {}".format(avg_start, avg_stop, enstrophy[(t/period > avg_start) & (t/period < avg_stop)].mean()))

    outfile = str(output_path.joinpath('{}_enstrophy.png'.format(basename)))
    fig_en.savefig(outfile)

def plot_velocities(velocities, t, period, basename, output_path='./', calc_growth_rate=False, growth_start=1, growth_stop=2):

    w_rms = velocities[-1]
    u_rms = velocities[0]
    if len(velocities) == 3:
        v_rms = velocities[1]
    else:
        v_rms = None


    print("period = {}".format(period))
    fig_vel = plt.figure(figsize=(16,8))
    ax = fig_vel.add_axes([0.1,0.1,0.8,0.8])
    ax.semilogy(t/period, w_rms, label=r"$w_{rms}$")
    if v_rms is not None:
        ax.semilogy(t/period, v_rms, label=r"$v_{rms}$")
    ax.semilogy(t/period, u_rms, label=r"$u_{rms}$")
    ax.set_ylabel("rms velocities", fontsize=20)
    ax.set_xlabel(r"$t/t_{1}$", fontsize=20)

    if calc_growth_rate:
        gamma_w, w0 = compute_growth(w_rms, t, period, growth_start, growth_stop)
        ax.semilogy(t/period, w0*np.exp(gamma_w*t), 'k-.', label='$\gamma_w/\Omega_1 = %f$' % (gamma_w*period/(2*np.pi)))
        print("Growth rate gamma_w = {:18.12e} in raw units".format(gamma_w))
    ax.legend(loc='lower right').draw_frame(False)

    outfile = str(output_path.joinpath('{}_rms_vel.png'.format(basename)))
    fig_vel.savefig(outfile)
    
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

    eta = float(args['--eta'])
    growth_start = float(args['--growth-start'])
    growth_stop = float(args['--growth-stop'])
    calc_growth = args['--calc-growth']

    avg_start = float(args['--avg-start'])
    avg_stop = float(args['--avg-stop'])
    
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


    omega1 = 1/eta - 1.
    period = 2*np.pi/omega1
    print("omega1 = {}".format(omega1))
    files = args['<files>']
    ts = read_timeseries(files)
    vels = [ts['u_rms'],]
    if 'v_rms' in ts:
        vels.append(ts['v_rms'])
    vels.append(ts['w_rms'])
    plot_velocities(vels, ts['time'], period, basename, output_path=output_path,calc_growth_rate=calc_growth,growth_start=growth_start,growth_stop=growth_stop)
    plot_energy(ts['KE'], ts['time'], period, basename, avg_start, avg_stop, output_path=output_path)
    if 'enstrophy' in ts:
        plot_enstrophy(ts['enstrophy'], ts['time'], period, basename, avg_start, avg_stop, output_path=output_path)

