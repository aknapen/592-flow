from examples.exp_configs.non_rl.traffic_light_grid import flow_params
from flow.core.experiment import Experiment


# # number of time steps
flow_params['env'].horizon = 3000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)
