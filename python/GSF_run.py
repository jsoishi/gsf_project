"""
Dedalus script for 2D/3D GSF simulations


Usage:
    GSF_run.py [--Re=<Re> --mu=<mu> --eta=<eta> --N2=<N2> --no-chi --tracer --Pr=<Pr> --Lz=<Lz>  --restart=<restart_file> --nr=<nr> --ntheta=<ntheta> --nz=<nz> --filter=<filter> --mesh=<mesh> --stop_sim_time=<stop_sim_time> --stop_wall_time=<stop_wall_time> --run_note=<run_note>] 

Options:
    --Re=<Re>      Reynolds number [default: 80]
    --mu=<mu>      mu [default: 0]
    --N2=<N2>      Brunt-Vaisala squared (in units of Omega1) [default: 1]
    --no-chi       switch off thermal diffusion [default: False]
    --tracer       switch on tracer field [default: False]
    --Pr=<Pr>      Prandtl Number [default: 0.3]
    --eta=<eta>    eta [default: 0.6925207756232687]
    --Lz=<Lz>      Lz  [default: 2.0074074463832545]
    --restart=<restart_file>   Restart from checkpoint
    --nr=<nr>                  radial r (Chebyshev) resolution [default: 32]
    --nz=<nz>                  vertical z (Fourier) resolution [default: 32]
    --ntheta=<ntheta>          azimuthal theta (Fourier) resolution [default: 0]
    --filter=<filter>          fraction of modes to keep in ICs [default: 0.5]
    --mesh=<mesh>              processor mesh (you're in charge of making this consistent with nproc) [default: None]
    --stop_sim_time=<stop_sim_time>       simulation stop time (in inner periods) [default: 15]
    --stop_wall_time=<stop_wall_time>     wall stop time (in hours) [default: inf]
    --run_note=<run_note>      tag to ID specific run (appended to output directory) [default: None]
"""
import glob
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
tracer = args['--tracer']
Pr  = float(args['--Pr'])
N2 = float(args['--N2'])
Lz = float(args['--Lz'])
filter_frac = float(args['--filter'])

nr = int(args['--nr'])
ntheta = int(args['--ntheta'])
nz = int(args['--nz'])

restart = args['--restart']
mesh = args['--mesh']

stop_sim_time = float(args['--stop_sim_time'])
stop_wall_time = args['--stop_wall_time']

if stop_wall_time == 'inf':
    stop_wall_time = np.inf
else:
    stop_wall_time = float(stop_wall_time)

if mesh == 'None':
    mesh = None
else:
    mesh = [int(i) for i in mesh.split(',')]

run_note = args['--run_note']
if run_note == 'None':
    run_note = None

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_re{0:5.02e}_mu{1:5.02e}_eta{2:5.02e}_Pr{3:5.02e}_N2{4:5.02e}_filter{5:5.02e}_nr{6:d}_ntheta{7:d}_nz{8:d}/".format(Re, mu, eta, Pr, N2, filter_frac,nr, ntheta,nz)
if nochi:
    data_dir = data_dir.strip("/")
    data_dir += "_nochi/"
if tracer:
    data_dir = data_dir.strip("/")
    data_dir += "_tracer/"
if run_note:
    data_dir = data_dir.strip("/")
    data_dir += "_{}/".format(run_note)

# if restart:
#     restart_dirs = glob.glob(data_dir+"restart*")
#     if restart_dirs:
#         restart_dirs.sort()
#         last = int(re.search("_restart(\d+)", restart_dirs[-1]).group(1))
#         data_dir += "_restart{}".format(last+1)
#     else:
#         if os.path.exists(data_dir):
#             data_dir += "_restart1"

from dedalus.tools.config import config

config['logging']['filename'] = os.path.join(data_dir,'dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
logger = logging.getLogger(__name__)
# root = logging.root
# for h in root.handlers:
#     h.setLevel("DEBUG")

from equations import GSF_boussinesq_equations
from filter_field import filter_field

if nochi:
    logger.warn("Overriding Pr!")
    Pr = 1e4

# configure GSF equations
GSF = GSF_boussinesq_equations(nr=nr, ntheta=ntheta, nz=nz,tracer=tracer, mesh=mesh)
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

logger.info("stopping at sim_time = {} periods or wall_time = {} hours".format(stop_sim_time, stop_wall_time))

logger.info("saving run in: {}".format(data_dir))

ts = de.timesteppers.RK443
solver= problem.build_solver(ts)

for k,v in problem.parameters.items():
    logger.info("problem parameter {}: {}".format(k,v))

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

    ## add perturbations to temperature
    T.set_scales(GSF.domain.dealias, keep_data=False)
    T['g'] = noise 
    if filter_frac != 1.: 
        logger.info("Beginning filter")
        filter_field(T,frac=filter_frac)
        logger.info("Finished filter")
    else:
        logger.warn("No filtering applied to ICs! This is probably bad!")
    T['g'] *= A0 * np.sin(np.pi*(r-r_in))
    T.differentiate('r',out=Tr)

    write = 0
    dt = 1e-3
    logger.info("Finished initalization")
else:
    logger.info("restarting from {}".format(restart))

    write, dt = solver.load_state(restart, -1)
    logger.info("starting from write {0:} with dt={1:10.5e}".format(write,dt))

omega1 = 1/GSF.eta - 1.
period = 2*np.pi/omega1

solver.stop_sim_time = stop_sim_time*period

if nochi:
    solver.stop_sim_time = 2.*period
solver.stop_wall_time = stop_wall_time * 3600.
solver.stop_iteration = np.inf

output_time_cadence = 0.1*period/4.
scalar_output_time_cadence = 0.1*output_time_cadence
analysis_tasks = GSF.initialize_output(solver, data_dir, sim_dt_profile=output_time_cadence, sim_dt_slice=output_time_cadence, sim_dt_scalar=scalar_output_time_cadence)
logger.info("Starting CFL")
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)
if GSF.threeD:
    CFL.add_velocities(('u', 'v', 'w'))
else:
    CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("abs(DivU)", name='divu')
flow.add_property("integ(r*KE)", name='KE')
flow.add_property("integ(r*enstrophy)", name='enstrophy')

if GSF.threeD:
    geo_factor = 1
else:
    geo_factor = 2*np.pi

dt = CFL.compute_dt()
logger.info("done CFL")
# Main loop
start_time = time.time()

while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time: %e, Inner rotation periods: %e, dt: %e' %(solver.iteration, solver.sim_time, solver.sim_time/period, dt))
        logger.info('Max |divu| = {}'.format(flow.max('divu')))
        logger.info('Total KE per Lz = {}'.format(geo_factor*flow.max('KE')/Lz))
        logger.info('Total enstrophy per Lz = {}'.format(geo_factor*flow.max('enstrophy')/Lz))
    dt = CFL.compute_dt()

# write last checkpoint
end_world_time = solver.get_world_time()
end_wall_time = end_world_time - solver.start_time
solver.evaluator.evaluate_handlers([analysis_tasks[0],], timestep = dt, sim_time = solver.sim_time, world_time=end_world_time, wall_time=end_wall_time, iteration=solver.iteration)

end_time = time.time()

# Print statistics
logger.info('Total wall time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))

logger.info('beginning join operation')

for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)

