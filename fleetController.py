from flow.networks import FigureEightNetwork
from flow.core.params import VehicleParams, NetParams, InitialConfig, SumoParams, EnvParams
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import GridRouter, ContinuousRouter
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs. import Figure, ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment

name = "fleet_controller_example"

vehicles = VehicleParams()

vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=4)


additional_params={
    'grid_array': {
        'row_num': 3,
        'col_num': 2,
        'inner_length': 500,
        'short_length': 500,
        'long_length': 500,
        'cars_top': 1,
        'cars_bot': 1,
        'cars_left': 2,
        'cars_right': 1,
    },
    'horizontal_lanes': 1,
    'vertical_lanes': 1,
    'speed_limit': {
        'vertical': 35,
        'horizontal': 35
    },
    'traffic_lights': True
}

net_params = NetParams(additional_params=additional_params)
initial_config = InitialConfig(spacing="uniform", perturbation=1)
traffic_lights = TrafficLightParams()

sim_params = SumoParams(sim_step=0.1, render=True, emission_path="sim_data")
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

flow_params = dict(
    exp_tag='fleet_controller_example',
    env_name=TrafficLightGridEnv,
    network=TrafficLightGridNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params['env'].horizon = 3000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=True)