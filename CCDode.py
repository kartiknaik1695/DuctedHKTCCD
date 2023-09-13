import numpy as np
import openmdao.api as om
from BEMcompu import Computation
from math import pi
from scipy.interpolate import interp2d


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
        
class TechnoEco(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):

        nn = self.options['num_nodes']
        # Inputs
        self.add_input('x', val=np.zeros(nn), desc='velocity', units='rad/s')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)
        self.add_input('Jdot', val=np.ones(nn))
        
        self.add_output('LCOE', val=0.5, desc='levelized cost of energy', units=None)
        # Setup partials
        arange = np.arange(nn)
        self.declare_partials(of="LCOE", wrt="x", method = 'fd')
        self.declare_partials(of="LCOE", wrt="u", method = 'fd')
        self.declare_partials(of="LCOE", wrt="Jdot", method = 'fd')
        
    def compute(self, inputs, outputs):
        
        om = inputs['x']
        u = inputs['u']
        # tau = inputs['torque']
        
        pwr = inputs['Jdot']
        
        nn = self.options['num_nodes']
        
        time = np.linspace(0, nn-1, num = nn)
        
        # Base failure rate
        lam_bf = 1e-4
        bf = 1 - np.exp(-time*lam_bf)
        
        # Find mean values for failure rate coefficients
        omM = np.mean(om)
        uM = np.mean(u)
        
        # Efficiency map
        # Effom = np.load('effX.npy')*(pi/30)
        # Effu = np.load('effY.npy')*(2.4)
        # Eff = np.load('Effi.npy')
        
        # f = interp2d(Effom, Effu, Eff, kind = 'linear')
        # netEff = (1/100)*f(omM, uM)
        netEff = 1
        
        omD = 25
        uD = 800
        Pmax = 5000
        
        # Failure rate coefficients
        Fom = 1 + np.power((omM/omD),0.7)
        Fu = np.power((0.5*uM/uD),4.69)
        
        F = Fom*Fu
        f_t = bf*F
        
        Pfail = np.sum(f_t*time)/np.sum(time)
        
        # Add normalized costs
        Ccap = 2.5*0.082
        Cop = 0.25*(Pfail/0.065)
        
        C = Ccap + Cop
        
        lcoe = np.sum(np.clip(pwr,0,Pmax))*((1-F)*netEff/C)
                
        outputs["LCOE"] = lcoe
        
    

class TurbineODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('torque_comp', Computation(num_nodes=nn), promotes_inputs=['x'])
        self.add_subsystem('ode_comp', Turbine(num_nodes=nn), promotes_inputs=['u', 'x'], promotes_outputs=['xdot', 'Jdot'])
        self.add_subsystem('lcoe_comp', TechnoEco(num_nodes=nn),promotes_inputs=['u', 'x','Jdot'], promotes_outputs=['LCOE'])
        self.connect('torque_comp.torque', 'ode_comp.torque')
        # self.connect('Jdot', 'lcoe_comp.Jdot')
