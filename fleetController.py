from flow.networks import TrafficLightGridNetwork
from flow.core.params import VehicleParams, NetParams, InitialConfig, SumoParams, EnvParams, TrafficLightParams, SumoCarFollowingParams
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import GridRouter
from flow.envs.fleet_control_env import FleetControlEnv, ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment

name = "fleet_controller_example"

# Define vehicle parameters
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        tau=1.1,
        max_speed=30,
        decel=7.5,
        accel=3,
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=4
)

# Define network parameters
additional_params = {
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

# Define simulation parameters
sim_params = SumoParams(
    sim_step=0.1,
    render=True,
    emission_path="sim_data"
)

# Define environment parameters
env_params = EnvParams(
    horizon=3000,
    additional_params=ADDITIONAL_ENV_PARAMS
)

# Define flow parameters
flow_params = dict(
    exp_tag='fleet_controller_example',
    env_name=FleetControlEnv,
    network=TrafficLightGridNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# Create and run the experiment
exp = Experiment(flow_params)
_ = exp.run(1, convert_to_csv=True)
