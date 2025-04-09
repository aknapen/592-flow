# from examples.exp_configs.rl.singleagent.flowagent import flow_params
from examples.exp_configs.rl.singleagent.flowidm import flow_params
from flow.core.experiment import Experiment


# # number of time steps
flow_params['env'].horizon = 45
exp = Experiment(flow_params)

# run the sumo simulation
print("Experiment running")
_ = exp.run(1,  convert_to_csv=True)
print("Experiment finished")