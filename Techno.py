import numpy as np
import openmdao.api as om
from scipy.interpolate import interp2d


class failureComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):

        nn = self.options['num_nodes']
        # Inputs
        self.add_input("time_phase", units='s', val=np.ones(nn))
        self.add_input('slope', units=None, val=1.0)
        self.add_output('Pfail', val=np.ones(nn), desc='derivative of base failure rate', units='1.0/s')

        # Setup partials
        arange = np.arange(nn)
        self.declare_partials(of="Pfail", wrt="time_phase", rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        time = inputs["time_phase"]
        a = inputs["slope"]
        outputs['Pfail'] = time+1/a*np.exp(-a*time)

    def compute_partials(self, inputs, partials):

        time = inputs["time_phase"]
        a = inputs["slope"]

        partials["Pfail", "time_phase"] = 1-np.exp(-a*time)


class TimeAverage(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('state_nodes', types=int,
                             desc='Number of state nodes to be evaluated in the RHS')

        self.options.declare('control_nodes', types=int,
                             desc='Number of control nodes to be evaluated in the RHS')

    def setup(self):

        n_state = self.options['state_nodes']
        n_control = self.options['control_nodes']
        # Inputs
        self.add_input('x', val=np.zeros(n_state), desc='velocity', units='rad/s')
        self.add_input('u', val=np.ones(n_control), desc='control', units=None)
        self.add_output('Fomega', val=1.0, desc='omega dependent failure', units=None)
        self.add_output('Fu', val=1.0, desc='u dependent failure', units=None)

        # Setup partials
        self.declare_partials(of="Fomega", wrt="x", method="fd")
        self.declare_partials(of="Fu", wrt="u", method="fd")

    def compute(self, inputs, outputs):

        om = inputs['x']
        u = inputs['u']

        # Find mean values for failure rate coefficients
        omM = np.mean(om)
        uM = np.mean(u)

        omD = 25
        uD = 800

        # Failure rate coefficients
        Fom = 1 + np.power((omM/omD),0.7)
        Fu = np.power((0.5*uM/uD),4.69)

        outputs["Fomega"] = Fom
        outputs["Fu"] = Fu


class TechnoEco(om.ExplicitComponent):

    def setup(self):

        # Inputs

        self.add_input('Fomega', val=1.0)
        self.add_input('Fu', val=1.0)
        self.add_input('Pfail', val=1.0)
        self.add_input('Egen', val=1.0)
        
        self.add_output('LCOE', val=0.5, desc='levelized cost of energy', units=None)
        # Setup partials
        self.declare_partials(of="LCOE", wrt="Fomega", method = 'fd')
        self.declare_partials(of="LCOE", wrt="Fu", method = 'fd')
        self.declare_partials(of="LCOE", wrt="Pfail", method = 'fd')
        self.declare_partials(of="LCOE", wrt="Egen", method = 'fd')
        
    def compute(self, inputs, outputs):

        Fom = inputs["Fomega"]
        Fu = inputs["Fu"]

        Egen = inputs["Egen"]
        Pfail = inputs["Pfail"]
        
        # Efficiency map
        # Effom = np.load('effX.npy')*(pi/30)
        # Effu = np.load('effY.npy')*(2.4)
        # Eff = np.load('Effi.npy')
        
        # f = interp2d(Effom, Effu, Eff, kind = 'linear')
        # netEff = (1/100)*f(omM, uM)
        netEff = 1
        
        
        F = Fom*Fu

        Pmax = 5000
        
        Pfail = inputs["Pfail"]
        
        # Add normalized costs
        C1 = 2.5*0.082
        C2 = 0.25*(Pfail/0.065)
        
        lcoe = (1-F)*netEff*Egen/(C1+C2*Pfail)
                
        outputs["LCOE"] = lcoe