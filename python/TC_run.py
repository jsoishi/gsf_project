"""
Dedalus script for 2D Taylor-Couette Flow


Usage:
    FC_nrho3.5.py [--Re=<Re> --mu=<mu> --eta=<eta> --Lz=<Lz>  --restart=<restart_file> --nz=<nz>] 

Options:
    --Re=<Re>      Reynolds number [default: 80]
    --mu=<mu>      mu [default: 0]
    --eta=<eta>    eta [default: 0.6925207756232687]
    --Lz=<Lz>      Lz  [default: 2.0074074463832545]
    --restart=<restart_file>   Restart from checkpoint
    --nz=<nz>                  vertical z (chebyshev) resolution [default: 128]
"""
import os
import numpy as np
import time 
import dedalus.public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)
root = logging.root
for h in root.handlers:
    h.setLevel("DEBUG")

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
