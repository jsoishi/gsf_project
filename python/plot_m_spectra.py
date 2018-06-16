"""
 make movies of u, v, w m spectra


Usage:
    plot_m_spectra.py <files>... [--eta=<eta> --output=<dir>]

Options:
    --eta=<eta>              eta = R1/R2 [default: .99]
    --output=<dir>           Output directory 
"""
import h5py
import dedalus.public as de
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def calc_m_spectra(data, field, index=-1):
    if field is None:
        raise ValueError("When using data from HDF5 files, you must specify a field.")
    field_name = 'tasks/{}c'.format(field)
    power = np.abs(data[field_name][index])

    m_power = power.sum(axis=2).sum(axis=0)

    return m_power

def plot_spectra(data, index, time):

    u_m_spectra = calc_m_spectra(data, 'u', index)
    v_m_spectra = calc_m_spectra(data, 'v', index)
    w_m_spectra = calc_m_spectra(data, 'w', index)

    m_max = int(np.ceil(len(u_m_spectra)/2))

    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax1.semilogy(u_m_spectra[:m_max].real, 'ko-')
    ax1.set_xlabel('$m$')
    ax1.set_ylabel('u power')

    ax2 = fig.add_subplot(132)
    ax2.semilogy(v_m_spectra[:m_max].real, 'ko-')
    ax2.set_xlabel('$m$')
    ax2.set_ylabel('v power')
    ax2.set_title(r'$t/t_1 = {:5.3f}$'.format(time))

    ax3 = fig.add_subplot(133)
    ax3.semilogy(w_m_spectra[:m_max].real, 'ko-')
    ax3.set_xlabel('$m$')
    ax3.set_ylabel('w power')

    fig.subplots_adjust()

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

    files = args['<files>']
    data_files = sorted(files, key=lambda x: int(os.path.split(x)[1].split('.')[0].split('_s')[1]))

    eta = float(args['--eta'])
    omega1 = 1/eta - 1.
    period = 2*np.pi/omega1

    count = 0
    for f in data_files:
        ts = h5py.File(f,'r')

        time = ts['scales']['sim_time'][:]/period

        for i in range(ts['tasks']['vc'][:].shape[0]):
            tt = time[i]
            fig = plot_spectra(ts, i, tt)
            print("saving spectra {:04d}".format(count))
            filen = str(output_path / "vel_m_spectra_{:04d}".format(count))
            fig.savefig(filen)
            plt.close()
            count += 1
        ts.close()
