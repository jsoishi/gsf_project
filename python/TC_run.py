"""
Dedalus script for 2D Taylor-Couette Flow


Usage:
    TC_run.py [--Re=<Re> --mu=<mu> --eta=<eta> --Lz=<Lz>  --restart=<restart_file> --nz=<nz>] 

Options:
    --Re=<Re>      Reynolds number [default: 80]
    --mu=<mu>      mu [default: 0]
    --eta=<eta>    eta [default: 0.6925207756232687]
    --Lz=<Lz>      Lz  [default: 2.0074074463832545]
    --restart=<restart_file>   Restart from checkpoint
    --nz=<nz>                  vertical z (Fourier) resolution [default: 128]
"""
import os
import numpy as np
import time 
import dedalus.public as de
from dedalus.extras import flow_tools
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing=True
except ImportError:
    print("No Checkpointing module. Setting checkpointing to false.")
    do_checkpointing=False

import logging
logger = logging.getLogger(__name__)
# root = logging.root
# for h in root.handlers:
#     h.setLevel("DEBUG")

from equations import TC_equations


from docopt import docopt
args = docopt(__doc__)

import sys

mu = float(args['--mu'])
Re = float(args['--Re'])
eta = float(args['--eta'])
Lz = float(args['--Lz'])

nz = int(args['--nz'])
nr = nz

restart = args['--restart']

# save data in directory named after script
data_dir = sys.argv[0].split('.py')[0]
data_dir += "_re{0:5.02e}_mu{1:5.02e}_eta{2:5.02e}/".format(Re, mu, eta)
logger.info("saving run in: {}".format(data_dir))

TC = TC_equations(nr=nr, nz=nz)
TC.set_parameters(mu, eta, Re, Lz)
TC.set_IVP_problem()
TC.set_BC()
problem = TC.problem

if TC.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

ts = de.timesteppers.RK443
solver= problem.build_solver(ts)


if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)

if restart is None:
     # Random perturbations, need to initialize globally
    gshape = TC.domain.dist.grid_layout.global_shape(scales=TC.domain.dealias)
    slices = TC.domain.dist.grid_layout.slices(scales=TC.domain.dealias)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    A0 = 1e-3

    # ICs
    u = solver.state['u']
    ur = solver.state['ur']
    v = solver.state['v']
    vr = solver.state['vr']
    w = solver.state['w']
    wr = solver.state['wr']
    r = TC.domain.grid(-1,scales=TC.domain.dealias)
    r_in = TC.R1

    v['g'] = TC.calc_v0()
    v.differentiate('r',out=vr)

    ## add perturbations
    phi = TC.domain.new_field(name='phi')
    phi.set_scales(TC.domain.dealias, keep_data=False)

    phi['g'] = A0 * noise
    phi.differentiate('r',out=u)
    u['g'] *= -1*np.sin(np.pi*(r - r_in))
    phi.differentiate('z',out=w)
    w['g'] *= np.sin(np.pi*(r - r_in))
    u.differentiate('r',out=ur)
    w.differentiate('r',out=wr)
else:
    logger.info("restarting from {}".format(restart))
    checkpoint.restart(restart, solver)


omega1 = problem.parameters['v_l']/r_in
period = 2*np.pi/omega1
solver.stop_sim_time = 15*period
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

output_time_cadence = 0.1*period
analysis_tasks = TC.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

