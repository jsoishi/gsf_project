"""
Dedalus script for 2D/3D GSF simulations


Usage:
    GSF_run.py [--Re=<Re> --mu=<mu> --eta=<eta> --N2=<N2> --no-chi --Pr=<Pr> --Lz=<Lz>  --restart=<restart_file> --nz=<nz>] 

Options:
    --Re=<Re>      Reynolds number [default: 80]
    --mu=<mu>      mu [default: 0]
    --N2=<N2>      Brunt-Vaisala squared (in units of Omega1) [default: 1]
    --no-chi       switch off thermal diffusion [default: False]
    --Pr=<Pr>      Prandtl Number [default: 0.3]
    --eta=<eta>    eta [default: 0.6925207756232687]
    --Lz=<Lz>      Lz  [default: 2.0074074463832545]
    --restart=<restart_file>   Restart from checkpoint
    --nz=<nz>                  vertical z (Fourier) resolution [default: 32]
"""
import logging
import os
import sys
import time 

import numpy as np
from docopt import docopt

# parse arguments
args = docopt(__doc__)

mu = float(args['--mu'])
Re = float(args['--Re'])
eta = float(args['--eta'])
nochi = args['--no-chi']
Pr  = float(args['--Pr'])
N2 = float(args['--N2'])
Lz = float(args['--Lz'])

nz = int(args['--nz'])
nr = nz

restart = args['--restart']

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_re{0:5.02e}_mu{1:5.02e}_eta{2:5.02e}_Pr{3:5.02e}_N2{4:5.02e}_nz{5:d}/".format(Re, mu, eta, Pr, N2, nz)
if nochi:
    data_dir = data_dir.strip("/")
    data_dir += "_nochi/"

from dedalus.tools.config import config

config['logging']['filename'] = os.path.join(data_dir,'dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
logger = logging.getLogger(__name__)

# use checkpointing if available
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing=True
except ImportError:
    logger.warn("No Checkpointing module. Setting checkpointing to false.")
    do_checkpointing=False

from equations import GSF_boussinesq_equations
from filter_field import filter_field

if nochi:
    logger.warn("Overriding Pr!")
    Pr = 1e4

# configure GSF equations
GSF = GSF_boussinesq_equations(nr=nr, nz=nz)
GSF.set_parameters(mu, eta, Re, Lz, Pr, N2)
GSF.set_IVP_problem()
GSF.set_BC()
problem = GSF.problem

if GSF.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

        # write any hg diffs to a text file
        if GSF.hg_diff:
            diff_filename = os.path.join(data_dir,'diff.txt')
            with open(diff_filename,'w') as file:
                file.write(GSF.hg_diff)

logger.info("saving run in: {}".format(data_dir))

ts = de.timesteppers.RK443
solver= problem.build_solver(ts)

for k,v in problem.parameters.items():
    logger.info("problem parameter {}: {}".format(k,v))

if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)

if restart is None:
     # Random perturbations, need to initialize globally
    gshape = GSF.domain.dist.grid_layout.global_shape(scales=GSF.domain.dealias)
    slices = GSF.domain.dist.grid_layout.slices(scales=GSF.domain.dealias)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    A0 = 1e-5

    # ICs
    v = solver.state['v']
    vr = solver.state['vr']
    T = solver.state['T']
    Tr = solver.state['Tr']
    r = GSF.domain.grid(-1,scales=GSF.domain.dealias)
    r_in = GSF.R1

    v['g'] = GSF.calc_v0()
    v.differentiate('r',out=vr)

    ## add perturbations to temperature
    T.set_scales(GSF.domain.dealias, keep_data=False)
    T['g'] = noise 
    filter_field(T,frac=0.5)
    T['g'] *= A0 * np.sin(np.pi*(r-r_in))
    T.differentiate('r',out=Tr)

    write = 0
    dt = 1e-3
else:
    logger.info("restarting from {}".format(restart))

    write, dt = solver.load_state(restart, -1)

omega1 = problem.parameters['v_l']/r_in
period = 2*np.pi/omega1

solver.stop_sim_time = 12.5*period

if nochi:
    solver.stop_sim_time = 2.*period
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

output_time_cadence = 0.01*period
analysis_tasks = GSF.initialize_output(solver, data_dir, sim_dt=output_time_cadence, write_num=write)

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.3,
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

