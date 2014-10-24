"""
Taylor-Couette flow 

Marcus Gellert's problematic run

"""

import numpy as np
import time
import shelve

import logging
logger = logging.getLogger(__name__)

from dedalus2.public import *
from dedalus2.data.field import Field
TC = ParsedProblem(axis_names=['z','θ','r'],
                    field_names=['p', 'u', 'v', 'w', 'ur', 'vr', 'wr'],
                    param_names=['nu', 'v_l', 'v_r', 'E', 'Phi'])

# equations
# TC.add_equation("ur + u/r + dθ(v)/r + dz(w) = 0")
# TC.add_equation("dt(u) - nu*dr(ur) - nu*ur/r - nu*dθ(dθ(u))/(r*r) - nu*dz(dz(u)) + nu*u/(r*r) + nu*2.*dθ(v)/(r*r) + dr(p) = -u*ur - v*dθ(u)/r - w*dz(u) + v*v/r")
# TC.add_equation("dt(v) - nu*dr(vr) - nu*vr/r - nu*dθ(dθ(v))/(r*r) - nu*dz(dz(v)) + nu*v/(r*r) - nu*2.*dθ(u)/(r*r) + dθ(p) = -u*vr - v*dθ(v)/r - w*dz(v) - u*v/r")
# TC.add_equation("dt(w) - nu*dr(wr) - nu*wr/r - nu*dθ(dθ(w))/(r*r) - nu*dz(dz(w)) + dz(p) = -u*wr - v*dθ(w)/r - w*dz(w)")
# TC.add_equation("ur - dr(u) = 0")
# TC.add_equation("vr - dr(v) = 0")
# TC.add_equation("wr - dr(w) = 0")

TC.add_equation("r*ur + u + dθ(v) + r*dz(w) = 0")
TC.add_equation("(r*r)*dt(u) - (r*r)*nu*dr(ur) - r*nu*ur - nu*dθ(dθ(u)) - (r*r)*nu*dz(dz(u)) + nu*u + nu*2.*dθ(v) + (r*r)*dr(p) = -u*ur*(r*r) - v*dθ(u)*r - w*dz(u)*(r*r) + v*v*r")
TC.add_equation("(r*r)*dt(v) - (r*r)*nu*dr(vr) - r*nu*vr - nu*dθ(dθ(v)) - (r*r)*nu*dz(dz(v)) + nu*v - nu*2.*dθ(u) + r*dθ(p) = -u*vr*(r*r) - v*dθ(v)*r - w*dz(v)*(r*r) - u*v*r")
TC.add_equation("(r*r)*dt(w) - (r*r)*nu*dr(wr) - r*nu*wr - nu*dθ(dθ(w)) - (r*r)*nu*dz(dz(w)) + (r*r)*dz(p) = -u*wr*(r*r) - v*dθ(w)*r - w*dz(w)*(r*r)")
TC.add_equation("ur - dr(u) = 0")
TC.add_equation("vr - dr(v) = 0")
TC.add_equation("wr - dr(w) = 0")

# boundary conditions
TC.add_left_bc("u = 0")
TC.add_left_bc("v = v_l")
TC.add_left_bc("w = 0")
TC.add_right_bc("u = 0", condition="dθ != 0 or dz != 0")
TC.add_right_bc("v = v_r")
TC.add_right_bc("w = 0")
TC.add_int_bc("p = 0", condition="dθ == 0 and dz == 0")
# tank parameters

eta = 0.5 # R1/R2
r_in = eta/(1. - eta)
r_out = 1./(1. - eta)
height = 4.
Rec = 115.1 
Re = 2. * Rec

# bases
r_basis = Chebyshev(33, interval=(r_in, r_out), dealias=2/3)
θ_basis = Fourier(32, interval=(0,2*np.pi), dealias=2/3)
z_basis = Fourier(128, interval=(0., height), dealias=2/3)
domain = Domain([z_basis, θ_basis, r_basis], 
                grid_dtype=np.float64, mesh=[1,4])

TC.parameters['nu'] = 1/Re
TC.parameters['v_l'] = 1.0
TC.parameters['v_r'] = 0.
#TC.parameters['E'] = 0.5 * (u*u + v*v + w*w)
#TC.parameters['E_merid'] = 0.5* (u*u + w*w)

TC.expand(domain, order=4)

ts = timesteppers.SBDF1
IVP = solvers.IVP(TC, domain, ts)


# initial conditions
phi = Field(domain, name='phi')
r = domain.grid(2)
θ = domain.grid(1)
z = domain.grid(0)

u = IVP.state['u']
v = IVP.state['v']
w = IVP.state['w']
ur = IVP.state['ur']
vr = IVP.state['vr']
wr = IVP.state['wr']
# irrotational perturbation, 'cause it's easier
phi['g'] = 1e-3 * np.random.randn(*v['g'].shape)
phi.differentiate(2,u)
u['g'] *= -1*np.sin(np.pi*(r - r_in))
phi.differentiate(0,w)
w['g'] *= np.sin(np.pi*(r - r_in))
u.differentiate(2,ur)
w.differentiate(2,wr)

# analytic TC solution 

v['g'] = eta/(1-eta**2) * (1./(r*(1-eta)) - r * (1.-eta)) 
v.differentiate(1,vr)


# import pylab as P
# P.imshow(u['g'])
# P.colorbar()
# P.savefig('init_u.png')
# P.clf()
# P.imshow(w['g'])
# P.colorbar()
# P.savefig('init_w.png')
# exit()

dt = max_dt = 0.1
omega1 = 1./r_in
period = 2*np.pi/omega1
IVP.stop_sim_time = 15.*period
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 100000

dx = domain.grid_spacing(0)
dy = domain.grid_spacing(1)
dz = domain.grid_spacing(2)
safety = 0.1
dt_cadence = 5

def cfl_dt(safety=1.):

    minut = np.min(np.abs(dx / IVP.state['u']['g'].real))
    minvt = np.min(np.abs(dy / IVP.state['v']['g'].real))
    minwt = np.min(np.abs(dz / IVP.state['w']['g'].real))
    dt = safety * min(minut, minvt, minwt)

    if domain.distributor.size > 1:
        dt = domain.distributor.comm_cart.gather(dt, root=0)
        if domain.distributor.rank == 0:
            dt = [min(dt)] * domain.distributor.size
        dt = domain.distributor.comm_cart.scatter(dt, root=0)

    return dt


# analysis


# Integrated energy every 10 iterations
analysis1 = IVP.evaluator.add_file_handler("scalar_data", iter=10)
analysis1.add_task("Integrate(0.5 * (u*u + v*v + w*w))", name="total kinetic energy")
analysis1.add_task("Integrate(0.5 * (u*u + w*w))", name="meridional kinetic energy")
analysis1.add_task("Integrate(0.5 * (v*v))", name="toridal kinetic energy")

# Snapshots every hour, create new file at 1GB
analysis2 = IVP.evaluator.add_file_handler('snapshots',sim_dt=0.5*period, max_writes=1, parallel=False)
analysis2.add_system(IVP.state, layout='g')

# radial profiles every 100 timestpes
analysis3 = IVP.evaluator.add_file_handler("radial_profiles", iter=100)
analysis3.add_task("Integrate(r*v, dz)", name='Angular Momentum')

# Initial Euler steps
presteps = 20
for i in range(presteps):
    IVP.step(dt/presteps)
    logger.info('Prestep Iteration: %i, Time: %e' %(IVP.iteration, IVP.sim_time))
    dt = min(max_dt, cfl_dt(safety))


#IVP.timestepper = timesteppers.CNAB2(TC.nfields, domain)
IVP.timestepper = timesteppers.RK443(TC.nfields, domain)
dt = min(max_dt, cfl_dt(safety))
# Main loop
start_time = time.time()

while IVP.ok:
    IVP.step(dt)
    logger.info('Iteration: %i, Time: %e, dt: %e' %(IVP.iteration, IVP.sim_time, dt))
    if IVP.iteration % dt_cadence == 0:
        dt = min(max_dt, cfl_dt(safety))

    # enforce Hermetian symmetry
    if IVP.iteration % 100 == 0 or IVP.iteration % 100 == 1:
        for f in IVP.state.fields:
            f.require_grid_space()

end_time = time.time()

# Print statistics
logger.info('Total time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %IVP.iteration)
logger.info('Average timestep: %f' %(IVP.sim_time/IVP.iteration))
