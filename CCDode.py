import numpy as np
import openmdao.api as om
from BEMcompu import Computation
from math import pi
from scipy.interpolate import interp2d
from Techno import failureComp


class Turbine(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):

        nn = self.options['num_nodes']
        # Inputs
        self.add_input('x', val=np.zeros(nn), desc='velocity', units='rad/s')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)
        self.add_input('torque', val=np.ones(nn))
        self.add_input('mass', val=234) # Inertia, assumed to be constant
        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='rad/s**2',
                         tags=['dymos.state_rate_source:x', 'dymos.state_units:rad/s'])
        self.add_output('Jdot', val=np.ones(nn), desc='derivative of objective', units='1.0/s')

        # Setup partials
        arange = np.arange(nn)
        self.declare_partials(of="xdot", wrt="torque", rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="u", rows=arange, cols=arange)
        self.declare_partials(of="Jdot", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="Jdot", wrt="torque", rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        x = inputs['x']
        u = inputs['u']
        torque = inputs['torque']
        m = inputs['mass']
        outputs['xdot'] = (torque - u) / m # Simplified 1st order ODE, xdot = (fluid-induced torque - generator load) / Inertia
        outputs['Jdot'] = torque * x # MechanicalPower =  torque * turbine rotating speed

    def compute_partials(self, inputs, partials):

        x = inputs["x"]
        m = inputs["mass"]
        u = inputs['u']
        torque = inputs["torque"]
        partials["xdot", "torque"] = 1.0 / m
        partials["xdot", "u"] = -1.0 / m

        partials["Jdot", "x"] = torque
        partials["Jdot", "torque"] = x
        
    

class TurbineODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('torque_comp', Computation(num_nodes=nn), promotes_inputs=['x'], promotes_outputs=["Pfail"])
        self.add_subsystem('ode_comp', Turbine(num_nodes=nn), promotes_inputs=['u', 'x'], promotes_outputs=['xdot', 'Jdot'])
        self.connect('torque_comp.torque', 'ode_comp.torque')
        # self.add_subsystem('failureComp', failureComp(num_nodes=nn), promotes_outputs=["failuredot"])
        # self.connect('Jdot', 'lcoe_comp.Jdot')
