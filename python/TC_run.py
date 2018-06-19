"""
Dedalus script for 2D Taylor-Couette Flow

Default parameters from Barenghi (1991, J. Comp. Phys.).


Usage:
    TC_run.py [--Re=<Re> --mu=<mu> --eta=<eta> --Lz=<Lz>  --restart=<restart_file> --nr=<nr> --ntheta=<ntheta> --nz=<nz> --filter=<filter> --mesh=<mesh> --m1=<m1>] 

Options:
    --Re=<Re>      Reynolds number [default: 80]
    --mu=<mu>      mu [default: 0]
    --eta=<eta>    eta [default: 0.6925207756232687]
    --Lz=<Lz>      Lz  [default: 2.0074074463832545]
    --restart=<restart_file>   Restart from checkpoint
    --nr=<nr>                  radial (Chebysh) resolution [default: 32]
    --ntheta=<ntheta>          aximuthal (Fourier) resolution [default: 0]
    --nz=<nz>                  vertical z (Fourier) resolution [default: 32]
    --filter=<filter>          fraction of modes to keep in ICs [default: 0.5]
    --mesh=<mesh>              processor mesh (you're in charge of making this consistent with nproc) [default: None]
    --m1=<m1>                  initial m perturbation. If this is non-zero, make axisymmetric noise with a single sin(m*theta) perturbation [default: 0]
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

nr = int(args['--nr'])
ntheta = int(args['--ntheta'])
nz = int(args['--nz'])

m1 = int(args['--m1'])
ic_axisymmetric = (m1 > 0)

mesh = args['--mesh']
if mesh == 'None':
    mesh = None
else:
    mesh = [int(i) for i in mesh.split(',')]


restart = args['--restart']

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_re{0:5.02e}_mu{1:5.02e}_eta{2:5.02e}_Lz{3:5.02e}_filter{4:5.02e}_nr{5:d}_ntheta{6:d}_nz{7:d}/".format(Re, mu, eta, Lz, filter_frac, nr, ntheta, nz)
logger.info("saving run in: {}".format(data_dir))

TC = TC_equations(nr=nr, ntheta=ntheta, nz=nz, mesh=mesh)
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
    # ICs
    u = solver.state['u']
    ur = solver.state['ur']
    v = solver.state['v']
    vr = solver.state['vr']
    w = solver.state['w']
    wr = solver.state['wr']
    r = TC.domain.grid(-1,scales=TC.domain.dealias)
    if TC.threeD:
        theta = TC.domain.grid(1,scales=TC.domain.dealias)
    r_in = TC.R1

    # Random perturbations, need to initialize globally
    gshape = TC.domain.dist.grid_layout.global_shape(scales=TC.domain.dealias)
    slices = TC.domain.dist.grid_layout.slices(scales=TC.domain.dealias)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)

    if TC.threeD:
        noise_r = rand.standard_normal(gshape)
        noise_z = rand.standard_normal(gshape)

    if ic_axisymmetric:
        logger.info("Making perturbations with only m1 = {}...".format(m1))
        slices_axi = [slices[0], 0, slices[-1]]
        noise = noise[slices_axi][:,None,:] * np.ones_like(noise[slices]) * np.sin(m1*theta)
        if TC.threeD:
            noise_r = noise_r[slices_axi][:,None,:] * np.ones_like(noise_r[slices]) * np.sin(m1*theta)
            noise_z = noise_z[slices_axi][:,None,:] * np.ones_like(noise_z[slices]) * np.sin(m1*theta)
    else:
        noise = noise[slices]
        if TC.threeD:
            noise_r = noise_r[slices]
            noise_z = noise_z[slices]
    A0 = 1e-3

    ## add perturbations
    ## g is the vector potential

    g_theta = TC.domain.new_field(name='g_theta')
    g_theta.set_scales(TC.domain.dealias, keep_data=False)
    if TC.threeD:
        g_r = TC.domain.new_field(name='g_r')
        g_z = TC.domain.new_field(name='g_z')

        g_r.set_scales(TC.domain.dealias, keep_data=False) 
        g_z.set_scales(TC.domain.dealias, keep_data=False) 

    g_theta['g'] = A0 * noise * np.sin(np.pi*(r - r_in))
    if TC.threeD:
        g_r['g'] = A0 * noise_r * np.sin(np.pi*(r - r_in))
        g_z['g'] = A0 * noise_z * np.sin(np.pi*(r - r_in))

    if filter_frac != 1.: 
        logger.info("Beginning filter")
        filter_field(g_theta,frac=filter_frac)
        if TC.threeD:
            filter_field(g_r, frac=filter_frac)
            filter_field(g_z, frac=filter_frac)
        logger.info("Finished filter")
    else:
        logger.warn("No filtering applied to ICs! This is probably bad!")

    g_theta.differentiate('z',out=u)
    u['g'] *= -1
    g_theta.differentiate('r',out=w)
    w['g'] += g_theta['g']/r
    if TC.threeD:
        u['g'] += g_z.differentiate('theta')['g']/r
        g_r.differentiate('z',out=v)
        v['g'] -= g_z.differentiate('r')['g']
        w['g'] -= g_r.differentiate('theta')['g']/r

    u.differentiate('r',out=ur)
    w.differentiate('r',out=wr)
    if TC.threeD:
        v.differentiate('r', out=vr)
else:
    logger.info("restarting from {}".format(restart))
    solver.load_state(restart, -1)


omega1 = 1/TC.eta - 1.
period = 2*np.pi/omega1
solver.stop_sim_time = 15*period
solver.stop_wall_time = 24*3600.#np.inf
solver.stop_iteration = np.inf

output_time_cadence = 0.1*period
analysis_tasks = TC.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

CFL = flow_tools.CFL(solver, initial_dt=1e-3, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)

if TC.threeD:
    CFL.add_velocities(('u', 'v', 'w'))
else:
    CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("abs(DivU)", name='divu')
flow.add_property("integ(r*KE)", name='KE')
flow.add_property("integ(r*enstrophy)", name='enstrophy')

dt = CFL.compute_dt()
# Main loop
start_time = time.time()

if TC.threeD:
    geo_factor = 1
else:
    geo_factor = 2*np.pi

logger.info("Starting main loop...")
while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time: %e, Inner rotation periods: %e, dt: %e' %(solver.iteration, solver.sim_time, solver.sim_time/period, dt))
        logger.info('Max |divu| = {}'.format(flow.max('divu')))
        logger.info('Total KE per Lz = {}'.format(geo_factor*flow.max('KE')/Lz))
        logger.info('Total enstrophy per Lz = {}'.format(geo_factor*flow.max('enstrophy')/Lz))
    dt = CFL.compute_dt()


end_time = time.time()

# Print statistics
logger.info('Total wall time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))

logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)

