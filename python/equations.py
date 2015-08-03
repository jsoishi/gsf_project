import numpy as np
import os
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Equations():
    def __init__(self):
        pass

    def set_IVP_problem(self, *args, **kwargs):
        self._set_domain()
        self.problem = de.IVP(self.domain, variables=self.variables)
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

    def initialize_output(self, solver ,data_dir, **kwargs):
        self.analysis_tasks = []
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("v", name="v")
        analysis_slice.add_task("w", name="w")
        #analysis_slice.add_task("theta", name="buoyancy")
        self.analysis_tasks.append(analysis_slice)
        
        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(KE)", name="KE")

        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(v_rms)", name="v_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(epicyclic_freq_sq)", name="epicyclic_freq_sq")
        #analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        
        self.analysis_tasks.append(analysis_profile)

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=20, parallel=False, **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        #analysis_scalar.add_task("vol_avg(PE)", name="PE")
        #analysis_scalar.add_task("vol_avg(KE + PE)", name="TE")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(v_rms)", name="v_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        #analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")

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
            self.problem.add_bc("integ(p,'r') = 0", condition="ntheta == 0 and nz == 0")
        else:
            self.problem.add_bc("integ(p,'r') = 0", condition="nz == 0")

        self.problem.add_bc("left(v) = v_l")
        self.problem.add_bc("right(v) = v_r")

        self.problem.add_bc("left(w) = 0")
        self.problem.add_bc("right(w) = 0")


    def _set_subs(self):
        """
        this implements the cylindrical del operators. 
        NB: ASSUMES THE EQUATION SET IS PREMULTIPLIED BY A POWER OF r (SEE BELOW)!!!

        Lap_s --> scalar laplacian
        Lap_r --> r component of vector laplacian
        Lap_t --> theta component of vector laplacian
        Lap_z --> z component of vector laplacian

        """
        self.problem.substitutions['vel_sum_sq'] = 'u**2 + v**2 + w**2'

        # NB: this problem assumes delta = R2 - R1 = 1 
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "z")'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        self.problem.substitutions['KE'] = 'vel_sum_sq/2'
        self.problem.substitutions['u_rms'] = 'sqrt(u*u)'
        self.problem.substitutions['v_rms'] = 'sqrt(v*v)'
        self.problem.substitutions['w_rms'] = 'sqrt(w*w)'
        self.problem.substitutions['Re_rms'] = 'sqrt(vel_sum_sq)*Lz/nu'
        self.problem.substitutions['epicyclic_freq_sq']  = 'dr(r*r*r*r*v*v)/(r*r*r)'
        # if self.threeD:
        #     self.problem.substitutions['enstrophy'] = '(dy(w) - v_z)**2 + (u_z- dx(w) )**2 + (dx(v) - dy(u))**2'
        # else:
        #     self.problem.substitutions['enstrophy'] = '(u_z - dx(w))**2'

        
        if self.threeD:
            # assume pre-multiplication by r*r
            self.problem.substitutions['Lap_s(f, f_r)'] = "r*r*dr(f_r) + r*f_r + dtheta(dtheta(f)) + r*r*dz(dz(f))"
            self.problem.substitutions['Lap_r'] = "Lap_s(u, ur) - u - 2*dtheta(v)"
            self.problem.substitutions['Lap_t'] = "Lap_s(v, vr) + 2*dtheta(u) - v"
            self.problem.substitutions['Lap_z'] = "Lap_s(w, wr)"
            self.problem.substitutions['UdotGrad_s(f, f_r)'] = "r*r*u*f_r + r*v*dtheta(f) + r*r*w*dz(f)"
            self.problem.substitutions['UdotGrad_r'] = "UdotGrad_s(u, ur) - r*v*v"
            self.problem.substitutions['UdotGrad_t'] = "UdotGrad_s(v, vr) - r*u*v"
            self.problem.substitutions['UdotGrad_z'] = "UdotGrad_s(w, wr)"
        else:
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

    def __init__(self, nr=32, ntheta=0, nz=32, grid_dtype=np.float64, dealias=3/2):
        self.nr = nr 
        self.ntheta = ntheta
        self.nz = nz
        if ntheta:
            self.threeD = True
        else:
            self.threeD = False

        self.grid_dtype = grid_dtype
        self.dealias = dealias

        self.equation_set = 'incompressible TC'
        self.variables = ['u','ur','v','vr','w','wr','p']

    def _set_domain(self):
        """

        """
        #try:
        t_bases = self._set_transverse_bases()
        r_basis = self._set_r_basis()
        #except AttributeError:
        #    raise AttributeError("You must set parameters before constructing the domain.")

        bases = t_bases + r_basis
        
        self.domain = de.Domain(bases, grid_dtype=self.grid_dtype)        
        
    def _set_transverse_bases(self):
        z_basis = de.Fourier(  'z', self.nz, interval=[0., self.Lz], dealias=self.dealias)
        trans_bases = [z_basis,]
        if self.threeD:
            theta_basis = de.Fourier('theta', self.ntheta, interval=[0., 2*np.pi], dealias=self.dealias)
            trans_bases += theta_basis

        return trans_bases

    def _set_r_basis(self):
        r_basis = de.Chebyshev('r', self.nr, interval=[self.R1, self.R2], dealias=3/2)
        
        return [r_basis,]

    def set_parameters(self, mu, eta, Re1, Lz):
        self.mu = mu 
        self.eta = eta
        self.Re1 = Re1
        self.Lz = Lz

        self.R1 = self.eta/(1. - self.eta)
        self.R2 = 1./(1-self.eta)
        self.Omega1 = 1/self.R1
        self.Omega2 = self.mu*self.Omega1
        self.nu = self.R1 * self.Omega1/self.Re1

        self._eqn_params = {}
        self._eqn_params['nu'] = self.nu
        self._eqn_params['v_l'] = self.R1*self.Omega1
        self._eqn_params['v_r'] = self.R2*self.Omega2
        self._eqn_params['Lz'] = self.Lz

    def calc_v0(self):
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
        self.problem.add_equation("r*r*dt(u) - nu*Lap_r + r*r*dr(p) = -UdotGrad_r")

    def set_mom_t(self):
        theta_mom = "r*r*dt(v) - nu*Lap_t"
        if self.threeD:
            theta_mom += " + r*r*dtheta(p)"
        theta_mom += " = -UdotGrad_t"

        self.problem.add_equation(theta_mom)

    def set_mom_z(self):
        self.problem.add_equation("r*dt(w) - nu*Lap_z + r*dz(p) = -UdotGrad_z")

    def set_aux(self):
        self.problem.add_equation("ur - dr(u) = 0")
        self.problem.add_equation("vr - dr(v) = 0")
        self.problem.add_equation("wr - dr(w) = 0")

    def set_energy(self):
        pass

class GSF_boussinesq_equations(TC_equations):
    def __init__(self):
        self.equation_set = 'Spiegel-Veronis Compressible Boussinesq'
        self.variables = ['u','ur','v','vr','w','wr','T','Tr','p']

    def set_parameters(self, mu, eta, Re1, Lz, Pm):
        super(GSF_boussinesq_equations, self).set_parameters(mu, eta, Re1, Lz)
        self.Pm = Pm
        self.chi = self.Pm/self.nu
        self._eqn_params['chi'] = self.chi

    def set_mom_r(self):
        self.problem.add_equation("r*r*dt(u) - nu*Lap_r + r*r*dr(p) - T = -UdotGrad_r")

    def set_energy(self):
        self.problem.add_equation("r*dt(T) - r*chi*dr(Tr) - chi*Tr - r*chi*dz(dz(T))  + r*N2*u = -r*u*Tr - r*w*dz(T)")
    
    def set_aux(self):
        super(GSF_boussinesq_equations, self).set_aux()
        self.problem.add_equation("Tr - dr(T) = 0")
        

    def set_BC(self):
        super(GSF_boussinesq_equations, self).set_BC()
        self.problem.add_bc("left(T) = 0")
        self.problem.add_bc("right(T) = 0")
