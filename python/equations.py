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

class TC_experiment():
    def __init__(self):
        pass

    def set_background(self):
        

class Equations():
    def __init__(self):
        pass

    def set_IVP_problem(self, *args, **kwargs):
        self.problem = de.IVP(self.domain, variables=self.variables)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, **kwargs):
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega')
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def _set_subs(self):
        self.problem.substitutions['Div(f)']

    def set_equations(self):
        self.set_continuity()
        self.set_momentum()
        self.set_energy()
        self.set_aux()

    def initialize_output(self, solver ,data_dir, **kwargs):
        self.analysis_tasks = []
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("v", name="v")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("theta", name="buoyancy")
        analysis_tasks.append(analysis_slice)
        
        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(PE)", name="PE")
        analysis_profile.add_task("plane_avg(KE + PE)", name="TE")

        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(v_rms)", name="v_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        
        analysis_tasks.append(analysis_profile)

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=20, parallel=False, **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        analysis_scalar.add_task("vol_avg(PE)", name="PE")
        analysis_scalar.add_task("vol_avg(KE + PE)", name="TE")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(v_rms)", name="v_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")

        analysis_tasks.append(analysis_scalar)

        # workaround for issue #29
        self.problem.namespace['enstrophy'].store_last = True

        return self.analysis_tasks

    def set_BC(self, v_l, v_r):
        self.problem.add_bc("left(u) = 0")
        self.problem.add_bc("left(v) = v_l")
        self.problem.add_bc("left(w) = 0")
        self.problem.add_bc("right(u) = 0", condition="ntheta != 0 or nz != 0")
        self.problem.add_bc("right(v) = v_r")
        self.problem.add_bc("right(w) = 0")
        self.problem.add_bc("integ(p,'r') = 0", condition="ntheta == 0 and nz == 0")

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
        
        if self.threeD:
            # assume pre-multiplication by r*r
            self.problem.substitutions['Lap_s(f, f_r)'] = "r*r*dr(f_r) + r*f_r + dtheta(dtheta(f)) + r*r*dz(dz(f))"
            self.problem.substitutions['Lap_r'] = "Lap_s(u, u_r) - u - 2*dtheta(v)"
            self.problem.substitutions['Lap_t'] = "Lap_s(v, v_r) + 2*dtheta(u) - v"
            self.problem.substitutions['Lap_z'] = "Lap_s(w, w_r)"
            self.problem.substitutions['UdotGrad_s(f, f_r)'] = "r*r*u*f_r + r*dtheta(f) + r*r*dz(f)"
            self.problem.substitutions['UdotGrad_r'] = "UdotGrad_s(u, u_r) - r*v**2"
            self.problem.substitutions['UdotGrad_t'] = "UdotGrad_s(v, v_r) - r*u*v"
            self.problem.substitutions['UdotGrad_z'] = "UdotGrad_s(w, w_r)"
        else:
            # assume pre-multiplication by r for scalars and w, r*r for r, theta vector components
            self.problem.substitutions['Lap_s(f, f_r)'] = "r*dr(f_r) + f_r + r*dz(dz(f))"
            self.problem.substitutions['Lap_r(f, f_r, t)'] = "r*Lap_s(f, f_r) - f"
            self.problem.substitutions['Lap_t(f, f_r, r)'] = "r*Lap_s(f, f_r) - f"
            self.problem.substitutions['Lap_z(f, f_r)'] = "Lap_s(f,f_r)"
            self.problem.substitutions['UdotGrad_s(f, f_r)'] = "r*u*f_r + r*dz(f)"
            self.problem.substitutions['UdotGrad_r'] = "r*UdotGrad_s(u, u_r) - r*v**2"
            self.problem.substitutions['UdotGrad_t'] = "r*UdotGrad_s(v, v_r) - r*u*v"
            self.problem.substitutions['UdotGrad_z'] = "UdotGrad_s(w, w_r)"

class TC_equations(Equations):
    def __init__(self):
        self.equation_set = 'incompressible TC'
        self.variables = ['u','u_r','v','v_r','w','w_z','T1','T1_z','p']

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
        self.variables = ['u','u_r','v','v_r','w','w_z','T','T_r','p']

    def set_mom_r(self):
        self.problem.add_equation("r*r*dt(u) - nu*Lap_r + r*r*dr(p) - T = -UdotGrad_r")

    def set_energy(self):
        self.problem.add_equation("r*dt(T) - r*chi*dr(T_r) - chi*T_r - r*chi*dz(dz(T)) -r*N2*u = -r*u*T_r - r*w*dz(T)")
    
    def set_aux(self):
        super(GSF_boussinesq_equations, self).set_aux()
        self.problem.add_equation("T_r - dr(T) = 0")
        

    def set_BC(self):
        super(GSF_boussinesq_equations, self).set_BC()
        self.problem.add_bc("left(T) = 0")
        self.problem.add_bc("right(T) = 0")
