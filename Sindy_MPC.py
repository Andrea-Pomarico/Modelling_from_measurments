# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:51:18 2024
In this coda, The SINDYc algoritm has been integrated as system identification 
for MPC controller. The first part of the code concerns the system 
identification by simulating the ODE and then using SINDYc to find a 
parsimonoius model.

The second part concerns the MPC to control the voltage at bus 2 in the
power grids, Bc has been used as control variable.

@author: andre
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps
import do_mpc
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import casadi as ca


import re
import io
import contextlib

#%% First Part: system identification SINDYc

# training
# Generating random step data for Bc

def generate_random_step_function_Bc(duration=30, max_value=0.2, min_value=-0.2):
    num_steps = np.random.randint(7, 10)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

# Generating random step data for Pd
def generate_random_step_function_Pd(duration=30, max_value=0.4, min_value=-0.4):
    num_steps = np.random.randint(8, 10)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

# Bc function using random step data
def Bc(t, times, values):
    if t < times[0]:
        return values[0]
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return values[i]
    return values[-1]

# Pd function using random step data
def Pd(t, times, values):
    if t < times[0]:
        return values[0]
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return values[i]
    return values[-1]


# Parameters
V1 = 1
M = 1
Dg = 0.1
Dl = 1
tau = 1
k = 0.25
xl = 0.5  # constant
alfa=0.5
Gd=1

duration_Bc = 30
times_Bc, values_Bc, num_steps_Bc = generate_random_step_function_Bc(duration=duration_Bc)
print(num_steps_Bc)
duration_Pd = 30
times_Pd, values_Pd, num_steps_Pd = generate_random_step_function_Pd(duration=duration_Pd)
print(num_steps_Pd)

# Dynamical system
def system(x, t):
    return [
        1/M * (alfa*Pd(t, times_Pd, values_Pd)+(1-alfa)*Gd*x[2]**2 - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
        x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - alfa*Pd(t, times_Pd, values_Pd)-(1-alfa)*Gd*x[2]**2),
        (1/tau) * (-x[2]**2 * (1/xl - Bc(t, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - alfa* k * Pd(t, times_Pd, values_Pd)-(1-alfa)*k*Gd*x[2]**2)
    ]

Pd0 = np.random.uniform(0, 0.7)
Pm = Pd0

def equations(x):
    return [
        1/M * (Pm - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
        x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - Pd0),
        (1/tau) * (-x[2]**2 * (1/xl - Bc(0, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - k * Pd0)
    ]

# Initial condition
x0 = [0, 0, 1]

x_0 = fsolve(equations, x0) #Initial point


start = 0
stop = 30
t_train_seconds=stop;
dt = 0.001

# Generate the time vector
t_train = np.arange(start, stop, dt)

#Solution ODE
solution = odeint(system, x_0, t_train) 

Pd_values_test = np.array([Pd(ti, times_Pd, values_Pd) for ti in t_train])
Bc_values_test = np.array([Bc(ti, times_Bc, values_Bc) for ti in t_train])
Results_matrix = np.column_stack((Pd_values_test, Bc_values_test, solution))  # prime due colonne gli input, ultime 3 output

# Plot
plt.plot(t_train, Results_matrix[:, -3:])
plt.xlabel('Time [s]')
plt.legend(['omega', 'delta', 'v'])
plt.grid()
plt.show()

plt.plot(t_train, Results_matrix[:, 0])
plt.xlabel('Time [s]')
plt.legend(['Pd'])
plt.grid()
plt.show()

plt.plot(t_train, Results_matrix[:, 1])
plt.xlabel('Time [s]')
plt.legend(['Bc'])
plt.grid()
plt.show()



# SINDYc
feature_names = ['x1', 'x2', 'x3', 'Pd', 'u']
opt = ps.STLSQ(threshold=0.1)

#Define feature library
poly_lib = ps.PolynomialLibrary(degree=2)
fourier_lib = ps.FourierLibrary(n_frequencies=1)
combined_lib = poly_lib + fourier_lib
model = ps.SINDy(feature_library=combined_lib, optimizer=opt, feature_names=feature_names)
u_values = np.vstack((Pd_values_test, Bc_values_test)).T

# Fit the model
model.fit(solution, t=dt, u=u_values)

# Function to get the model as a string (for MPC later)
def sindy_print_to_str(model):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        model.print()
    return f.getvalue()

# Function to process the output of SINDy and format it
def process_sindy_output(sindy_output):
    # Regex is used to process the equations
    equations = re.findall(r'\((\w+)\)\' = (.+)', sindy_output)
    processed_equations = []
    for var, expr in equations:
        expr = expr.replace('^', '**')
        expr = re.sub(r'\s+', ' ', expr)  
        expr = re.sub(r'(\d+\.?\d*)\s+(\w+)', r'\1*\2', expr) 
        expr = re.sub(r'(\w+)\s+(\w+)', r'\1*\2', expr)  
        expr = re.sub(r'(\w+)\s*\(', r'\1*(', expr)  
        expr = re.sub(r'((?<=[+\-*/\s])-\s*)(\d+(\.\d+)?)', r'\1(\2)', expr) 
        expr = re.sub(r'^(-\s*)(\d+(\.\d+)?)', r'\1(\2)', expr)  
        expr = expr.replace (" ", "") 
        expr = expr.replace( "(1*x", "(x") 
        expr = expr.replace( "(1*u", "(u") 
        expr = expr.replace( "(1*Pd", "(Pd") 
        expr = expr.replace("+-", "-")
        processed_equations.append(f'd{var} = {expr}') 
    return processed_equations

sindy_output = sindy_print_to_str(model)
print(sindy_output)

processed = process_sindy_output(sindy_output)
for eq in processed:
    print(eq)

coefficients=model.coefficients()
basis_functions = combined_lib.get_feature_names()



# Simulation with SINDy to compare with ODE integration
x_sindy = model.simulate(solution[0], t_train, u=u_values)

t_train1 = t_train[1:]
plt.figure()
plt.plot(t_train, solution[:, 0], label='$\omega$')
plt.plot(t_train, solution[:, 1], label='$\delta$')
plt.plot(t_train, solution[:, 2], label='$v$')
plt.plot(t_train1, x_sindy[:, 0], '--', label='SINDy-$\omega$')
plt.plot(t_train1, x_sindy[:, 1], '--', label='SINDy-$\delta$')
plt.plot(t_train1, x_sindy[:, 2], '--', label='SINDy-$v$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('t [s]')
plt.grid(True)
plt.show()

#%% TESTING
# Frstly, we run a new model with new random inputs and the we used
# the SINDYc model found before to compare the result
def generate_random_step_function_Bc(duration=60, max_value=0.2, min_value=-0.2):
    num_steps = np.random.randint(7, 10)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

def generate_random_step_function_Pd(duration=60, max_value=0.4, min_value=-0.4):
    num_steps = np.random.randint(8, 10)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

def Bc(t, times, values):
    if t < times[0]:
        return values[0]
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return values[i]
    return values[-1]

def Pd(t, times, values):
    if t < times[0]:
        return values[0]
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return values[i]
    return values[-1]


V1 = 1
M = 1
Dg = 0.1
Dl = 1
tau = 1
k = 0.25
xl = 0.5  

duration_Bc = 60
times_Bc, values_Bc, num_steps_Bc = generate_random_step_function_Bc(duration=duration_Bc)
print(num_steps_Bc)
duration_Pd = 60
times_Pd, values_Pd, num_steps_Pd = generate_random_step_function_Pd(duration=duration_Pd)
print(num_steps_Pd)

def system(x, t):
    return [
        1/M * (alfa*Pd(t, times_Pd, values_Pd)+ (1-alfa)*Gd*x[2]**2 - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
        x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - alfa*Pd(t, times_Pd, values_Pd)-(1-alfa)*Gd*x[2]**2),
        (1/tau) * (-x[2]**2 * (1/xl - Bc(t, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - alfa* k * Pd(t, times_Pd, values_Pd)-(1-alfa)*k*Gd*x[2]**2)
    ]

Pd0 = np.random.uniform(0, 0.7)
Pm = Pd0

def equations(x):
    return [
        1/M * (Pm - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
        x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - Pd0),
        (1/tau) * (-x[2]**2 * (1/xl - Bc(0, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - k * Pd0)
    ]

x0 = [0, 0, 1]

x_0 = fsolve(equations, x0)

start = 0
stop = 60
t_train_seconds=stop;
dt = 0.001

t_train = np.arange(start, stop, dt)

#Simulate again ODE (different from Training)
solution = odeint(system, x_0, t_train)

Pd_values_test = np.array([Pd(ti, times_Pd, values_Pd) for ti in t_train])
Bc_values_test = np.array([Bc(ti, times_Bc, values_Bc) for ti in t_train])

Results_matrix = np.column_stack((Pd_values_test, Bc_values_test, solution))  # prime due colonne gli input, ultime 3 output

plt.plot(t_train, Results_matrix[:, -3:])
plt.xlabel('Time [s]')
plt.legend(['omega', 'delta', 'v'])
plt.grid()
plt.show()

plt.plot(t_train, Results_matrix[:, 0])
plt.xlabel('Time [s]')
plt.legend(['Pd'])
plt.grid()
plt.show()

plt.plot(t_train, Results_matrix[:, 1])
plt.xlabel('Time [s]')
plt.legend(['Bc'])
plt.grid()
plt.show()

u_values_test = np.vstack((Pd_values_test, Bc_values_test)).T

# Testing SINDYc model
x_sindy_test = model.simulate(solution[0], t_train, u=u_values_test)

t_train1 = t_train[1:]

plt.figure()
plt.plot(t_train, solution[:, 0], label='$\omega$')
plt.plot(t_train, solution[:, 1], label='$\delta$')
plt.plot(t_train, solution[:, 2], label='$v$')
plt.plot(t_train1, x_sindy_test[:, 0], '--', label='SINDy-$\omega$')
plt.plot(t_train1, x_sindy_test[:, 1], '--', label='SINDy-$\delta$')
plt.plot(t_train1, x_sindy_test[:, 2], '--', label='SINDy-$v$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('t [s]')
plt.grid(True)
plt.show()

combined_term = alfa * Pd_values_test + (1 - alfa) * Gd * solution[:, 2]**2

plt.figure()
plt.plot(t_train, combined_term, label=r'P_tot')
plt.xlabel('Time [s]')
plt.ylabel('P_tot[pu]')
plt.legend()
plt.grid(True)
plt.show()

#%%MPC
# Here, the SINDYc + MPC is tested

def string_to_casadi(expr_str, variables):
    var_dict = {var.name(): var for var in variables}

    math_functions = {
        'sin': casadi.sin,
        'cos': casadi.cos,
        'tan': casadi.tan,
        'exp': casadi.exp,
        'log': casadi.log,
        'sqrt': casadi.sqrt
    }

    symbol_dict = {**var_dict, **math_functions}

    for var_name, var_symbol in var_dict.items():
        expr_str = expr_str.replace(var_name, f"symbol_dict['{var_name}']")

    for func_name in math_functions.keys():
        expr_str = expr_str.replace(f"{func_name}*(", f"symbol_dict['{func_name}'](")
        expr_str = expr_str.replace(f"{func_name}(", f"symbol_dict['{func_name}'](")

    expr = eval(expr_str, {"casadi": casadi, "symbol_dict": symbol_dict}, symbol_dict)

    return expr



M = 1    
Dg = 0.1   
Dl = 1.0   
V1 = 1   
xl = 0.5   
tau = 0.5  
k = 0.25

Pd_0 = 0.2
Pm = Pd_0
Bc = 0
xl_initial = 0.5
n_horizon = 60

def initial_conditions(x):
    return [
        1/M * (Pm - (V1 * x[2] / xl_initial) * np.sin(x[1]) - Dg * x[0]),
        x[0] - (1/Dl) * ((V1 * x[2] / xl_initial) * np.sin(x[1]) - Pd_0),
        (1/tau) * (-x[2]**2 * (1/xl_initial - Bc) + (V1 * x[2] / xl_initial) * np.cos(x[1]) - k * Pd_0)
    ]

x00 = [0, 0, 1]
x0 = fsolve(initial_conditions, x00)

model_type = 'continuous'  
model = do_mpc.model.Model(model_type)

x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))
x3 = model.set_variable(var_type='_x', var_name='x3', shape=(1,1))

u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

Pd = model.set_variable(var_type='_tvp', var_name='Pd')

variables = [x1, x2, x3, Pd, u]

i = 1
for eq in processed:
        # Extract the right-hand side of the equation
        rhs = eq.split('=')[1].strip()

        # Convert string to CasADi expression
        casadi_expr = string_to_casadi(rhs, variables)

        # Set right hand side
        model.set_rhs(f'x{i}', casadi_expr)
        i += 1


model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 60,
    't_step': 0.05,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

u0 = np.array([[0]])  

x_ref = np.array([x0[0], x0[1], 1.0])  

lterm = 0*(x1 - x_ref[0])**2 + 0*(x2 - x_ref[1])**2 + 100*(x3 - x_ref[2])**2 + 0.1*u**2  # Tracking + penalità ingresso
mterm = 0*(x1 - x_ref[0])**2 + 0*(x2 - x_ref[1])**2 + 1000*(x3 - x_ref[2])**2  # Tracking finale

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=1e-2)  

mpc.bounds['lower', '_u', 'u'] = -1
mpc.bounds['upper', '_u', 'u'] = 2


tvp_template = mpc.get_tvp_template()

def tvp_fun(t_now):
    for k in range(n_horizon+1):
        if t_now < 20:
            tvp_template['_tvp', k, 'Pd'] = 0.2
        elif t_now < 40:
            tvp_template['_tvp', k, 'Pd'] = 0.7
        else:
            tvp_template['_tvp', k, 'Pd'] = 0.5

    return tvp_template

mpc.set_tvp_fun(tvp_fun)

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=setup_mpc['t_step'])
tvp_template1 = simulator.get_tvp_template()

def tvp_fun1(t_now):
        if t_now < 20:
            tvp_template1['Pd'] = 0.2
        elif t_now < 40:
            tvp_template1['Pd'] = 0.7
        else:
            tvp_template1['Pd'] = 0.5

        return tvp_template1

simulator.set_tvp_fun(tvp_fun1)
simulator.setup()

x0 = fsolve(initial_conditions, x00)
mpc.x0 = x0
simulator.x0 = x0

states = [x0]
controls = [u0.flatten()]

mpc.set_initial_guess()

n_steps = 1200

for k in range(n_steps):
    u0 = mpc.make_step(x0)

    x0 = simulator.make_step(u0)

    states.append(x0.flatten())
    controls.append(u0.flatten())

states = np.array(states) 
controls = np.array(controls)  

fig, ax = plt.subplots(4, 1, figsize=(10, 8))

ax[0].plot(states[:, 0], label='x1 (velocità)')
ax[0].set_ylabel('x1')
ax[0].legend()

ax[1].plot(states[:, 1], label='x2 (angolo)')
ax[1].set_ylabel('x2')
ax[1].legend()

ax[2].plot(states[:, 2], label='x3 (tensione)')
ax[2].set_ylabel('x3')
ax[2].legend()

ax[3].plot(controls, label='u (controllo)')
ax[3].set_ylabel('u')
ax[3].set_xlabel('Tempo')
ax[3].legend()


plt.tight_layout()
