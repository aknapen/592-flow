import numpy as np
from flow.envs.base import Env

from gym.spaces import Tuple, Dict, MultiDiscrete, Discrete
from gym.spaces.box import Box

ADDITIONAL_ENV_PARAMS = {
    # Number of vehicles in the fleet
    'num_vehicles': 3,
    # Maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # Maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # Minimum x-value in the network
    'min_x': 0,
    # Maximum x-value in the network
    'max_x': 500,
    # Minimum y-value in the network
    'min_y': 0,
    # Maximum y-value in the network
    'max_y': 500
}

class FleetControlEnv(Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        # Check that all necessary additional environmental parameters have 
        # been specified
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p)
                )
        
        super.__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        # Agent can apply acceleration within [-max_decel, max_accel] to each vehicle
        accel_space = Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles,),
            dtype=np.float32)
        
        # Agent can update the route of the vehicle by changing the direction of travel.
        # [0 = up, 1 = right, 2 = down, 3 = left] (clockwise winding)
        route_space = MultiDiscrete([4]*self.initial_vehicles.num_rl_vehicles)

        return Tuple((accel_space, route_space))
    
    @property
    def observation_space(self):        
        # The agent's observation space consists of a triplet of (velocity, x position, y position)
        # for each vehicle in the fleet:
        #   1. the vehicle's velocity (normalized to network's max velocity)
        #   2. the vehicle's x position in the network
        #   3. the vehicle's y position in the network
        velocity_space = Box(
            low=0,
            high=1,
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)
        
        x_space = Box(low=self.env_params.additional_params['min_x'], 
                      high=self.env_params.additional_params['max_x'], 
                      shape=(self.initial_vehicles.num_rl_vehicles,), 
                      dtype=np.uint32)
        
        y_space = Box(low=self.env_params.additional_params['min_y'], 
                      high=self.env_params.additional_params['max_y'], 
                      shape=(self.initial_vehicles.num_rl_vehicles,), 
                      dtype=np.uint32)
                         
        return Tuple(velocity_space, x_space, y_space)
    
    def get_updated_route(route_actions):
        pass

    def _apply_rl_actions(self, rl_actions):
        ids = self.k.vehicle.get_rl_ids()
        accel_actions = [action[0] for action in rl_actions]
        route_actions = [action[1] for action in rl_actions]

        # Update any vehicle routes
        # TODO: Need to figure out how to map 0,1,2,3 values to new routes
        routes = get_updated_route(route_actions)
        self.k.vehicle.choose_routes(ids, routes)

        # Update any vehicle accelerations
        self.k.vehicle.apply_acceleration(ids, accel_actions)
    
    def compute_distance_traveled(curr_positions, prev_positions):
        # Calculate element-wise Euclidean distances of each vehicle
        distances = np.sqrt(np.pow(curr_positions[0] - prev_positions[0], 2) + 
                            np.pow(curr_positions[1], prev_positions[1], 2))
        
        # Return cumulative distance for the fleet
        return np.sum(distances)
    
    def compute_reward(self, rl_actions, **kwargs):
        ids = self.k.vehicle.get_rl_ids()
        # Get current (x,y) positions for each vehicle
        positions = np.array(self.k.vehicle.get_2d_position(id) for id in ids)

        # Emission reward is MPG metric (does this need to be scaled by time?)
        distance_traveled = compute_distance_traveled(positions, self.prev_positions)
        fuel_consumed = np.sum(np.array([self.k.vehicle.get_fuel_consumption(id) for id in ids]))
        emission_rewards = distance_traveled / fuel_consumed

        # update prev positions
        self.prev_positions = positions

        # This will be a list of booleans where the ith element is true if positions[i] = destinations[i]
        destinations_reached = [(positions[id][0] == self.destinations[id][0] and 
                                 positions[id][1] == self.destinations[id][1]) for id in ids]
        # Agent accumulates -1 rewards for each vehicle that is not at its destination
        route_reward = np.sum(np.array([1 if destinations_reached(id) else -1 for id in ids]))

        return self.emission_weight * emission_rewards + self.route_weight * route_reward

    def get_state(self):
        ids = self.k.vehicle.get_rl_ids()
        speeds = [self.k.vehicle.get_speed(id) / self.k.network.max_speed() for id in ids]
        pos = [self.k.vehicle.get_2d_position(id) for id in ids]

        return np.array(speeds + pos)