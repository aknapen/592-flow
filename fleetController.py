from examples.exp_configs.rl.singleagent.flowagent import flow_params
from flow.core.experiment import Experiment


# # number of time steps
flow_params['env'].horizon = 3
exp = Experiment(flow_params)

# run the sumo simulation
print("Experiment running")
_ = exp.run(1)
print("Experiment finished")