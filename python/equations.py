import numpy as np
import os
import subprocess
from mpi4py import MPI

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Equations():
    def __init__(self):
        self.hg_diff = None

    def set_IVP_problem(self, *args, **kwargs):
        self._set_domain()
        self.problem = de.IVP(self.domain, variables=self.variables)
        self.problem.meta[:]['r']['dirichlet'] = True
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, **kwargs):
        self._set_domain()
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega')
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def _apply_params(self):
        for k,v in self._eqn_params.items():
            self.problem.parameters[k] = v

    def set_equations(self, *args, **kwargs):
        self._apply_params()
        self._set_subs()
        self.set_aux()
        self.set_continuity()
        self.set_momentum()
        self.set_energy()
        self.set_tracer()

    def initialize_output(self, solver, data_dir, sim_dt_slice=None, sim_dt_profile=None, sim_dt_scalar=None, **kwargs):
        self.analysis_tasks = []
        wall_dt_checkpoints = 3540. # 59 minutes
        checkpoints = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), max_writes=1, wall_dt=wall_dt_checkpoints)
        checkpoints.add_system(solver.state)
        self.analysis_tasks.append(checkpoints)

        if sim_dt_slice:
            analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, sim_dt=sim_dt_slice, **kwargs)
        else:
            analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("v", name="v")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("u", name="uc", layout="c")
        analysis_slice.add_task("v", name="vc", layout="c")
        analysis_slice.add_task("w", name="wc", layout="c")
        if 'T' in self.variables:
            analysis_slice.add_task("T", name="T")
        self.analysis_tasks.append(analysis_slice)

        if sim_dt_profile:
            analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, sim_dt=sim_dt_profile, **kwargs)
        else:
            analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(KE_fluct)", name="KE_fluct")
        analysis_profile.add_task("plane_avg(v_tot)", name="v_tot")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(v_rms)", name="v_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(epicyclic_freq_sq)", name="epicyclic_freq_sq")
        
        self.analysis_tasks.append(analysis_profile)

        if sim_dt_scalar:
            analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", parallel=False, sim_dt=sim_dt_scalar, **kwargs)
        else:
            analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", parallel=False, **kwargs)
        analysis_scalar.add_task("integ(r*KE)", name="KE")
        analysis_scalar.add_task("integ(r*KE_fluct)", name="KE_fluct")
        analysis_scalar.add_task("integ(KE_zonal_u)", name="KE_fluct_zonal_u")
        analysis_scalar.add_task("integ(KE_zonal_v)", name="KE_fluct_zonal_v")
        analysis_scalar.add_task("integ(KE_zonal_w)", name="KE_fluct_zonal_w")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(v_rms)", name="v_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        if self.threeD:
            analysis_scalar.add_task("probe(w)", name="w_probe")
        analysis_scalar.add_task("integ(r*enstrophy)", name="enstrophy")

        self.analysis_tasks.append(analysis_scalar)

        # workaround for issue #29
        #self.problem.namespace['enstrophy'].store_last = True

        return self.analysis_tasks

    def set_BC(self):
        self.problem.add_bc("left(u) = 0")
        if self.threeD:
            self.problem.add_bc("right(u) = 0", condition="ntheta != 0 or nz != 0")
        else:
            self.problem.add_bc("right(u) = 0", condition="nz != 0")
        if self.threeD:
            self.problem.add_bc("right(p) = 0", condition="ntheta == 0 and nz == 0")
        else:
            self.problem.add_bc("right(p) = 0", condition="nz == 0")

        self.problem.add_bc("left(v) = 0")
        self.problem.add_bc("right(v) = 0")

        self.problem.add_bc("left(w) = 0")
        self.problem.add_bc("right(w) = 0")
        
        if self.tracer:
            self.problem.add_bc("left(c) = 0")
            self.problem.add_bc("right(c) = 0")


    def _set_subs(self):
        """
        this implements the cylindrical del operators. 
        NB: ASSUMES THE EQUATION SET IS PREMULTIPLIED BY A POWER OF r (SEE BELOW)!!!

        Lap_s --> scalar laplacian
        Lap_r --> r component of vector laplacian
        Lap_t --> theta component of vector laplacian
        Lap_z --> z component of vector laplacian

        """
        self.problem.substitutions['A'] = '(1/eta - 1.)*(mu-eta**2)/(1-eta**2)'
        self.problem.substitutions['B'] = 'eta*(1-mu)/((1-eta)*(1-eta**2))'

        self.problem.substitutions['v0'] = 'A*r + B/r'
        self.problem.substitutions['dv0dr'] = 'A - B/(r*r)'

        self.problem.substitutions['v_tot'] = 'v0 + v'
        self.problem.substitutions['vel_sum_sq'] = 'u**2 + v_tot**2 + w**2'

        # NB: this problem assumes delta = R2 - R1 = 1 
        self.problem.substitutions['zonal_avg(A)'] = 'integ(A,"z")/Lz'
        if self.threeD:
            self.problem.substitutions['plane_avg(A)'] = 'integ(r*A,"z","theta")/(Lz*2*pi*r)'
            self.problem.substitutions['vol_avg(A)']   = 'integ(r*A)/(pi*(R2**2 - R1**2)*Lz)'
            self.problem.substitutions['probe(A)'] = 'interp(A,r={}, theta={}, z={})'.format(self.R1 + 0.5, 0., self.Lz/2.)
        else:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "z")/Lz'
            self.problem.substitutions['vol_avg(A)']   = 'integ(r*A)/Lz'
        self.problem.substitutions['KE'] = '0.5*vel_sum_sq'
        self.problem.substitutions['KE_fluct'] = '0.5*(u**2+v**2+w**2)'
        self.problem.substitutions['KE_zonal_u'] = '0.5*zonal_avg(u)**2'
        self.problem.substitutions['KE_zonal_v'] = '0.5*zonal_avg(v)**2'
        self.problem.substitutions['KE_zonal_w'] = '0.5*zonal_avg(w)**2'
        self.problem.substitutions['u_rms'] = 'sqrt(u*u)'
        self.problem.substitutions['v_rms'] = 'sqrt(v*v)'
        self.problem.substitutions['w_rms'] = 'sqrt(w*w)'
        self.problem.substitutions['Re_rms'] = 'sqrt(vel_sum_sq)*Lz/nu'
        self.problem.substitutions['epicyclic_freq_sq']  = 'dr(r*r*v*v)/(r*r*r)'
        if self.threeD:
            self.problem.substitutions['enstrophy'] = '0.5*((dtheta(w)/r - dz(v_tot))**2 + (dz(u) - wr )**2 + (vr + dv0dr + v_tot/r - dtheta(u))**2)'
        else:
            self.problem.substitutions['enstrophy'] = '0.5*(dz(v_tot)**2 + (dz(u) - wr)**2 + (vr + dv0dr + v_tot/r)**2)'

        if self.threeD:
            # not pre-multiplied...don't use this in an equation!
            self.problem.substitutions['DivU'] = "ur + u/r + dtheta(v)/r + dz(w)"
            # assume pre-multiplication by r*r
            self.problem.substitutions['Lap_s(f, f_r)'] = "r*r*dr(f_r) + r*f_r + dtheta(dtheta(f)) + r*r*dz(dz(f))"
            self.problem.substitutions['Lap_r'] = "Lap_s(u, ur) - u - 2*dtheta(v)"
            self.problem.substitutions['Lap_t'] = "Lap_s(v, vr) - v + 2*dtheta(u)"
            self.problem.substitutions['Lap_z'] = "Lap_s(w, wr)"
            self.problem.substitutions['UdotGrad_s(f, f_r)'] = "r*r*u*f_r + r*v*dtheta(f) + r*r*w*dz(f)"
            self.problem.substitutions['UdotGrad_r'] = "UdotGrad_s(u, ur) - r*v*v"
            self.problem.substitutions['UdotGrad_t'] = "UdotGrad_s(v, vr) + r*u*v"
            self.problem.substitutions['UdotGrad_z'] = "UdotGrad_s(w, wr)"
        else:
            # not pre-multiplied...don't use this in an equation!
            self.problem.substitutions['DivU'] = "ur + u/r + dz(w)"
            # assume pre-multiplication by r for scalars and w, r*r for r, theta vector components
            self.problem.substitutions['Lap_s(f, f_r)'] = "r*dr(f_r) + f_r + r*dz(dz(f))"
            self.problem.substitutions['Lap_r'] = "r*Lap_s(u, ur) - u"
            self.problem.substitutions['Lap_t'] = "r*Lap_s(v, vr) - v"
            self.problem.substitutions['Lap_z'] = "Lap_s(w,wr)"
            self.problem.substitutions['UdotGrad_s(f, f_r)'] = "r*u*f_r + r*w*dz(f)"
            self.problem.substitutions['UdotGrad_r'] = "r*UdotGrad_s(u, ur) - r*v*v"
            self.problem.substitutions['UdotGrad_t'] = "r*UdotGrad_s(v, vr) + r*u*v"
            self.problem.substitutions['UdotGrad_z'] = "UdotGrad_s(w, wr)"

class TC_equations(Equations):
    """
    delta = R2 - R1
    mu = Omega2/Omega1
    eta = R1/R2

    scale [L] = delta
    scale [T] = delta/(R1 Omega1)
    scale [V] = R1 Omega1
    """

    def __init__(self, nr=32, ntheta=0, nz=32, grid_dtype=np.float64, dealias=3/2, tracer=False, mesh=None):
        super(TC_equations,self).__init__()
        self.nr = nr 
        self.ntheta = ntheta
        self.nz = nz
        if ntheta:
            self.threeD = True
        else:
            self.threeD = False
        if mesh:
            self.mesh = mesh

        self.grid_dtype = grid_dtype
        self.dealias = dealias

        self.tracer = tracer

        self.equation_set = 'incompressible TC'
        self.variables = ['u','ur','v','vr','w','wr','p']
        if self.tracer:
            self.variables += ['c','cr']

    def _set_domain(self):
        """

        """
        #try:
        t_bases = self._set_transverse_bases()
        r_basis = self._set_r_basis()
        #except AttributeError:
        #    raise AttributeError("You must set parameters before constructing the domain.")

        bases = t_bases + r_basis
        if self.threeD and self.mesh:
            self.domain = de.Domain(bases, grid_dtype=self.grid_dtype,mesh=self.mesh)        
        else:
            self.domain = de.Domain(bases, grid_dtype=self.grid_dtype)        
        
    def _set_transverse_bases(self):
        z_basis = de.Fourier(  'z', self.nz, interval=[0., self.Lz], dealias=self.dealias)
        trans_bases = [z_basis,]
        if self.threeD:
            theta_basis = de.Fourier('theta', self.ntheta, interval=[0., 2*np.pi], dealias=self.dealias)
            trans_bases.append(theta_basis)

        return trans_bases

    def _set_r_basis(self):
        r_basis = de.Chebyshev('r', self.nr, interval=[self.R1, self.R2], dealias=3/2)
        
        return [r_basis,]

    def set_parameters(self, mu, eta, Re1, Lz, Sc=1):
        self.mu = mu 
        self.eta = eta
        self.Re1 = Re1
        self.Lz = Lz
        self.Sc = Sc

        self.R1 = self.eta/(1. - self.eta)
        self.R2 = 1./(1-self.eta)
        self.Omega1 = 1/self.R1
        self.Omega2 = self.mu*self.Omega1
        self.nu = self.R1 * self.Omega1/self.Re1

        #dye diffusivity 
        self.nu_dye = self.nu/self.Sc

        self._eqn_params = {}
        self._eqn_params['nu'] = self.nu
        self._eqn_params['nu_dye'] = self.nu_dye
        self._eqn_params['eta'] = self.eta 
        self._eqn_params['mu'] = self.mu
        self._eqn_params['Lz'] = self.Lz
        self._eqn_params['R1'] = self.R1
        self._eqn_params['R2'] = self.R2
        self._eqn_params['pi'] = np.pi

    def calc_v0(self):
        """Calculate the couette flow velocity on the grid.

        """
        r = self.domain.grid(-1)
        A = -self.Omega1 * self.eta**2 * (1-self.mu/self.eta**2)/(1-self.eta**2)
        B = self.Omega1 * self.R1**2 * (1 - self.mu)/((1-self.eta**2))
        v0 = A*r + B/r
        return v0

    def set_continuity(self):
        if self.threeD:
            self.problem.add_equation("r*ur + u + dtheta(v) + r*dz(w) = 0")
        else:
            self.problem.add_equation("r*ur + u + r*dz(w) = 0")

    def set_momentum(self):
        self.set_mom_r()
        self.set_mom_t()
        self.set_mom_z()

    def set_mom_r(self):
        r_mom = "r*r*dt(u) - nu*Lap_r - 2*r*v0*v"
        if self.threeD:
            r_mom += "+ r*v0*dtheta(u)"
        r_mom += "+ r*r*dr(p) = r*v0*v0 - UdotGrad_r"

        self.problem.add_equation(r_mom)

    def set_mom_t(self):
        theta_mom = "r*r*dt(v) - nu*Lap_t + r*r*dv0dr*u + r*v0*u"
        if self.threeD:
            theta_mom += "+ r*v0*dtheta(v) + r*dtheta(p)"
        theta_mom += " = -UdotGrad_t"

        self.problem.add_equation(theta_mom)

    def set_mom_z(self):
        if self.threeD:
            self.problem.add_equation("r*r*dt(w) - nu*Lap_z + r*r*dz(p) + r*v0*dtheta(w) = -UdotGrad_z")
        else:
            self.problem.add_equation("  r*dt(w) - nu*Lap_z +   r*dz(p)                  = -UdotGrad_z")

    def set_aux(self):
        self.problem.add_equation("ur - dr(u) = 0")
        self.problem.add_equation("vr - dr(v) = 0")
        self.problem.add_equation("wr - dr(w) = 0")
        if self.tracer:
            self.problem.add_equation("cr - dr(c) = 0")

    def set_energy(self):
        pass

    def set_tracer(self):
        if self.tracer:
            if self.threeD:
                self.problem.add_equation("r*r*dt(c) - nu_dye*Lap_s(c, cr) = -UdotGrad_s(c, cr)")
            else:
                self.problem.add_equation("r*dt(c) - nu_dye*Lap_s(c, cr) = -UdotGrad_s(c, cr)")

class GSF_boussinesq_equations(TC_equations):
    def __init__(self, *args, **kwargs):
        super(GSF_boussinesq_equations,self).__init__(*args, **kwargs)
        self.equation_set = 'Spiegel-Veronis Compressible Boussinesq'
        self.variables = ['u','ur','v','vr','w','wr','T','Tr','p']
        if self.tracer:
            self.variables += ['c','cr']

    def set_parameters(self, mu, eta, Re1, Lz, Pr, N2, Sc=1):
        super(GSF_boussinesq_equations, self).set_parameters(mu, eta, Re1, Lz, Sc=Sc)
        self.N2 = N2*self.Omega1**2
        self.Pr = Pr
        self.chi = self.nu/self.Pr

        self._eqn_params['chi'] = self.chi
        self._eqn_params['N2'] = self.N2

    def set_mom_r(self):
        r_mom = "r*r*dt(u) - nu*Lap_r - r*r*T - 2*r*v0*v"
        if self.threeD:
            r_mom += "+ r*v0*dtheta(u)"
        r_mom += "+ r*r*dr(p) = r*v0*v0 - UdotGrad_r"

        self.problem.add_equation(r_mom)

    def set_energy(self):
        if self.threeD:
            energy = "r*r*dt(T) - chi*Lap_s(T,Tr)  + r*r*N2*u + v0*dtheta(T) = -UdotGrad_s(T, Tr)"
        else:
            energy = "  r*dt(T) - chi*Lap_s(T,Tr)  +   r*N2*u                = -UdotGrad_s(T, Tr)"
        
        self.problem.add_equation(energy)
    
    def set_aux(self):
        super(GSF_boussinesq_equations, self).set_aux()
        self.problem.add_equation("Tr - dr(T) = 0")
        

    def set_BC(self):
        super(GSF_boussinesq_equations, self).set_BC()
        self.problem.add_bc("left(T) = 0")
        self.problem.add_bc("right(T) = 0")
