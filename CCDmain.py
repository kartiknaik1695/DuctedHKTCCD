import os
import openmdao.api as om
import dymos as dm
import numpy as np
from CCDode import TurbineODE
import matplotlib.pyplot as plt
import timeit
from curvature import curvature
from Techno import TimeAverage, TechnoEco

# Define a Dymos problem
prob = om.Problem()
prob.model = om.Group()
prob.driver = om.pyOptSparseDriver()
# Try ScipyOptimizer if pyOptSparse is not available
# prob.driver = om.ScipyOptimizeDriver()

# SNOPT settings
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.opt_settings['Major iterations limit'] = 5000
prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
prob.driver.opt_settings['Iterations limit'] = 200000

# Recorder & Output files
outputDir = "output_13"

if not os.path.exists(os.path.join(os.getcwd(), outputDir)):
    os.mkdir(outputDir)

recorder = om.SqliteRecorder('cases.sql')
prob.driver.opt_settings["Print file"] = os.path.join(outputDir, "SNOPT_print.out")
prob.driver.opt_settings["Summary file"] = os.path.join(outputDir, "SNOPT_summary.out")
prob.add_recorder(recorder)
prob.driver.add_recorder(recorder)

# Define control trajectories
Tsteps = 60
num_seg = Tsteps
traj = prob.model.add_subsystem('traj', dm.Trajectory())
phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=TurbineODE,
                                transcription=dm.GaussLobatto(num_segments=num_seg,
                                                              order=3,
                                                              compressed=True)))
# Define states in a CCD problem
# x: turbine rotating speed (rad/s)
# J: Output energy (J)
# u: generator load (Nm), control varaible
phase.add_state('x', fix_initial=False, lower=0,
                rate_source='xdot',
                units=None,
                targets='x', ref=1)  # target required because x0 is an input
phase.add_state('J', fix_initial=True, fix_final=False, lower=0,
                rate_source='Jdot',
                units=None, ref=1e5)

phase.add_state('Pfail', fix_initial=True, fix_final=False, rate_source='failuredot',
                units=None)

phase.add_control(name='u', units=None, lower=0, upper=800, continuity=True,
                  rate_continuity=True, targets='u', ref=0.5e3)

# Define design parameters
# theta(degree): twist angle along radius
# chord(m): chord length along radius

phase.add_parameter('xdata', targets=['torque_comp.xdata'], static_target=True, units=None,
                    val=np.array([0, 0.944, 0.913, 0.869, 1, 1]),
                    opt=True, lower=np.array([-4.0, 0.91, 0.91, 0.91, 0.55, 0.55]), 
                    upper=np.array([4.0, 0.99, 0.99, 0.99, 1.45, 1.45]), ref=0.5)


# Define time options
t_final = Tsteps
phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=t_final, units='s', targets='torque_comp.time_phase')

prob.model.add_subsystem("curvature", curvature(), promotes_outputs=["curvature"])
prob.model.add_constraint("curvature", lower = 0)
prob.model.add_subsystem("averageBase", TimeAverage(num_nodes=num_seg))
prob.model.add_subsystem("lcoe", TechnoEco(), promotes_outputs=["LCOE"])

prob.model.connect("traj.phase0.states:J", "lcoe.Egen", src_indices=[-1])
prob.model.connect("traj.phase0.states:Pfail", "lcoe.Pfail", src_indices=[-1])
prob.model.connect("traj.phase0.states:x", "averageBase.x")
prob.model.connect("traj.phase0.controls:u", "averageBase.u")
prob.model.connect("averageBase.Fomega", "lcoe.Fomega")
prob.model.connect("averageBase.Fu", "lcoe.Fu")

# Define objective function
# ref = -1.0: maximize J
# phase.add_objective('J', loc='final', ref=-1.0e5)
# phase.add_objective('LCOE', loc='final', ref=-1.0e5)
# phase.add_objective('LCOE', ref=-1)
phase.add_timeseries_output('torque_comp.torque')
# phase.add_timeseries_output('lcoe_comp.LCOE')
prob.model.add_objective("")
prob.setup(check=True)

# Problem initialization
prob['traj.phase0.t_initial'] = 0.0
prob['traj.phase0.t_duration'] = t_final

prob['traj.phase0.states:x'] = phase.interp(ys=[13.689, 13.689], nodes='state_input')
prob['traj.phase0.states:J'] = phase.interp(ys=[0, 100000], nodes='state_input')
prob['traj.phase0.controls:u'] = phase.interp(ys=[800, 800], nodes='control_input')


start = timeit.default_timer()
# Run problem & record data
dm.run_problem(prob)
J = prob.compute_totals(driver_scaling=True, return_format='flat_dict')
prob.set_solver_print(0)
prob.record("final_state")

stop = timeit.default_timer()

print('Time: ', stop - start)  


#%%
# A simple example on data visualizations
JJJ = prob.get_val('traj.phase0.timeseries.states:J')
TTT = prob.get_val('traj.phase0.timeseries.time')
XXX = prob.get_val('traj.phase0.timeseries.states:x')
UUU = prob.get_val('traj.phase0.timeseries.controls:u')
TTTORQUE = prob.get_val('traj.phase0.timeseries.torque')

xdata = prob.get_val('traj.phase0.timeseries.parameters:xdata')
xdata_final = xdata[0]
print('xDataFinal=', xdata_final)



fig = plt.figure()
cmap = plt.get_cmap("tab10")
# plt.plot(TT, JJ, color=cmap(0), marker="o")
plt.plot(TTT, JJJ, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Output Energy')
plt.savefig(os.path.join(outputDir, "OutputEnergyRAMPCCD.png"))

fig2 = plt.figure()
cmap = plt.get_cmap("tab10")
# plt.plot(TT, XX, color=cmap(0), marker="o")
plt.plot(TTT, XXX, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Turbine rotating speed (rad/s)')
plt.savefig(os.path.join(outputDir, "TURBINESPEEDRAMPCCD.png"))

plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(TTT, UUU, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Control Force')
plt.savefig(os.path.join(outputDir, "CONTROLLOADRAMPCCD.png"))


Uinf = np.sin(0.25 * TTT) * 0.2 + 1.5
Rtip = 0.511*1.4/0.7593546006

TSR = XXX*Rtip/Uinf

plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(TTT, TSR, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('TSR')










