"""
Dedalus script for 2D Taylor-Couette Flow

Default parameters from Barenghi (1991, J. Comp. Phys.).


Usage:
    TC_run.py [--Re=<Re> --mu=<mu> --eta=<eta> --Lz=<Lz>  --restart=<restart_file> --nz=<nz> --filter=<filter>] 

Options:
    --Re=<Re>      Reynolds number [default: 80]
    --mu=<mu>      mu [default: 0]
    --eta=<eta>    eta [default: 0.6925207756232687]
    --Lz=<Lz>      Lz  [default: 2.0074074463832545]
    --restart=<restart_file>   Restart from checkpoint
    --nz=<nz>                  vertical z (Fourier) resolution [default: 32]
    --filter=<filter>          fraction of modes to keep in ICs [default: 0.5]
"""
import os
import numpy as np
import time 
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

import logging
logger = logging.getLogger(__name__)
# root = logging.root
# for h in root.handlers:
#     h.setLevel("DEBUG")

from equations import TC_equations
from filter_field import filter_field


from docopt import docopt
args = docopt(__doc__)

import sys

mu = float(args['--mu'])
Re = float(args['--Re'])
eta = float(args['--eta'])
Lz = float(args['--Lz'])
filter_frac = float(args['--filter'])

nz = int(args['--nz'])
ntheta = 0
nr = nz

restart = args['--restart']

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_re{0:5.02e}_mu{1:5.02e}_eta{2:5.02e}_filter{3:5.02e}_nr{4:d}_ntheta{5:d}_nz{6:d}/".format(Re, mu, eta, filter_frac, nr, ntheta, nz)
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

for k,v in problem.parameters.items():
    logger.info("paramter {}: {}".format(k,v))

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
    phi['g'] = noise
    if filter_frac != 1.: 
        logger.info("Beginning filter")
        filter_field(phi,frac=filter_frac)
        logger.info("Finished filter")
    else:
        logger.warn("No filtering applied to ICs! This is probably bad!")

    phi.differentiate('r',out=u)
    u['g'] *= -1*np.sin(np.pi*(r - r_in))
    phi.differentiate('z',out=w)
    w['g'] *= np.sin(np.pi*(r - r_in))
    u.differentiate('r',out=ur)
    w.differentiate('r',out=wr)
else:
    logger.info("restarting from {}".format(restart))
    solver.load_state(restart, -1)


omega1 = problem.parameters['v_l']/r_in
period = 2*np.pi/omega1
solver.stop_sim_time = 50*period
solver.stop_wall_time = np.inf
solver.stop_iteration = 200#np.inf

output_time_cadence = 0.1*period
analysis_tasks = TC.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

CFL = flow_tools.CFL(solver, initial_dt=1e-3, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)
CFL.add_velocities(('u', 'w'))

dt = CFL.compute_dt()
# Main loop
start_time = time.time()

while solver.ok:
    solver.step(dt)
    logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
    dt = CFL.compute_dt()


end_time = time.time()

# Print statistics
logger.info('Total time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))

logger.info('beginning join operation')
if do_checkpointing:
    logger.info(data_dir+'/checkpoint/')
    post.merge_analysis(data_dir+'/checkpoint/')

for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)

