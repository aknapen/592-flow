import re
import numpy as np
from flow.envs.base import Env

from gym.spaces import Tuple, Dict, MultiDiscrete, Discrete
from gym.spaces.box import Box
from flow.networks.traffic_light_grid import ADDITIONAL_NET_PARAMS

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
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Initialize vehicle destinations (randomly set for now)
        self.destinations = {}
        # Initialize previous positions
        self.prev_positions = {}
        
        # Weights for different components of the reward function
        self.emission_weight = 1.0
        self.route_weight = 1.0

    @property
    def action_space(self):
        # Agent can apply acceleration within [-max_decel, max_accel] to each vehicle
        accel_space = Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            ######################################################################################
            shape=(self.initial_vehicles.num_rl_vehicles,),
            dtype=np.float32)
        
        # Agent can update the route of the vehicle by changing the direction of travel.
        # [0 = up, 1 = right, 2 = down, 3 = left] (clockwise winding)
        # The route space will be dynamic based on proximity to intersections
        # We need to determine the route space for each vehicle
        # Initially, we'll set up MultiDiscrete([4]) for each vehicle, but we'll handle
        # the actual route options in the _apply_rl_actions method
        ########################################################################################
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
    
    def get_updated_route(self, ids, route_actions):
    
        routes = []
        
        
        for i, veh_id in enumerate(ids):
            # Get current edge and position of the vehicle
            # this function maybe wrong/ get_route
            current_edge = self.k.vehicle.get_edge(veh_id)
            
            veh_pos = self.k.vehicle.get_2d_position(veh_id)
            
            near_intersection = False
            allowed_actions = []
            
            # Map of actions to direction names for route calculations
            direction_map = {0: "top", 1: "bot", 2: "right",  3: "left"}
            
            if hasattr(self.k.network, "_inner_nodes"):
                intersections = self.k.network._inner_nodes
                
                # nodes.append({
                #     "id": "center{}".format(row * self.col_num + col),
                #     "x": col * self.inner_length,
                #     "y": row * self.inner_length,
                #     "type": node_type,
                #     "radius": self.inner_nodes_radius
                # })

                # calculate distance from vehicle to each intersection
                for node in intersections:
                    distance = np.sqrt(
                        (veh_pos[0] - node["x"])**2 + 
                        (veh_pos[1] - node["y"])**2
                    )
                    
                    # near intersection radius means the vehicle is near the intersection
                    if distance < node["radius"]:
                        near_intersection = True
                        allowed_actions = [0, 1, 2, 3]  
                        break
                    
                    
            if not near_intersection:
                # check if the edge starts with bot{}_{} or top{}_{}
                if current_edge.startswith("bot") or current_edge.startswith("top"):
                    # Horizontal road - can only go left or right (1 or 3)
                    allowed_actions = [0, 1]  
                else:
                    allowed_actions = [2, 3]  

            action = route_actions[i]

            if action not in allowed_actions:
                # If invalid action, maintain current route (keep going in same direction)
                action = allowed_actions[0]
            
            current_route = self.k.vehicle.get_route(veh_id)
            match = re.match(r"(bot|right|top|left)(\d+)_(\d+)", current_edge)
            row, col = int(match.group(2)), int(match.group(3))
            direction = match.group(1)
            
            # Calculate new route based on action if near intersection
            if near_intersection:
                # current_edge looks like bot{row}_{col}, or right{row}_{col}
                #the numbers have to be the index of the current edge , current_edge looks like bot0_1
                #############################################################################################
                
                if current_edge.startswith("bot"):
                    if direction_map[action] == 'top':
                        new_edge = "top{}_{}".format(row, col)
                    elif direction_map[action] == 'bot':
                        new_edge = "bot{}_{}".format(row, col + 1)
                    elif direction_map[action] == 'right':
                        new_edge = "right{}_{}".format(row + 1, col)
                    elif direction_map[action] == 'left':
                        new_edge = "left{}_{}".format(row, col)
                elif current_edge.startswith("top"):
                    if direction_map[action] == 'top':
                        new_edge = "top{}_{}".format(row, col - 1)
                    elif direction_map[action] == 'bot':
                        new_edge = "bot{}_{}".format(row, col)
                    elif direction_map[action] == 'right':
                        new_edge = "right{}_{}".format(row +1, col - 1)
                    elif direction_map[action] == 'left':
                        new_edge = "left{}_{}".format(row, col - 1)
                elif current_edge.startswith("right"):
                    if direction_map[action] == 'top':
                        new_edge = "top{}_{}".format(row, col)
                    elif direction_map[action] == 'bot':
                        new_edge = "bot{}_{}".format(row, col + 1)
                    elif direction_map[action] == 'right':
                        new_edge = "right{}_{}".format(row + 1, col)
                    elif direction_map[action] == 'left':
                        new_edge = "left{}_{}".format(row, col)
                elif current_edge.startswith("left"):
                    if direction_map[action] == 'top':
                        new_edge = "top{}_{}".format(row - 1, col)
                    elif direction_map[action] == 'bot':
                        new_edge = "bot{}_{}".format(row - 1, col + 1)
                    elif direction_map[action] == 'right':
                        new_edge = "right{}_{}".format(row, col)
                    elif direction_map[action] == 'left':
                        new_edge = "left{}_{}".format(row - 1, col)   
            else:
                if direction_map[action].startswith(direction):
                    new_edge = current_edge
                else:
                    if direction == "bot" and direction_map[action] == "top":
                        new_edge = "top{}_{}".format(row, col)
                    elif direction == "top" and direction_map[action] == "bot":
                        new_edge = "bot{}_{}".format(row, col)
                    elif direction == "right" and direction_map[action] == "left":
                        new_edge = "left{}_{}".format(row, col)
                    elif direction == "left" and direction_map[action] == "right":
                        new_edge = "right{}_{}".format(row, col)
                        
                routes.append(new_edge)
            
        return routes

    def _apply_rl_actions(self, rl_actions):
        ids = self.k.vehicle.get_rl_ids()
        accel_actions = rl_actions[0]  # First element of tuple is acceleration actions
        route_actions = rl_actions[1]  # Second element of tuple is route actions

        # Update any vehicle routes
        routes = self.get_updated_route(ids, route_actions)
        self.k.vehicle.choose_routes(ids, routes)

        # Update any vehicle accelerations
        self.k.vehicle.apply_acceleration(ids, accel_actions)
    
    def compute_distance_traveled(self, curr_positions, prev_positions):
        # Calculate element-wise Euclidean distances of each vehicle
        distances = np.sqrt(np.power(curr_positions[0] - prev_positions[0], 2) + 
                           np.power(curr_positions[1] - prev_positions[1], 2))
        
        # Return cumulative distance for the fleet
        return np.sum(distances)
    
    def compute_reward(self, rl_actions, **kwargs):
        ids = self.k.vehicle.get_rl_ids()
        # Get current (x,y) positions for each vehicle
        positions = np.array([self.k.vehicle.get_2d_position(id) for id in ids])

        # Initialize previous positions if they don't exist
        if not self.prev_positions:
            self.prev_positions = positions
            return 0  # Return 0 reward on first step
            
        # Initialize destinations if they don't exist
        if not self.destinations:
            # Randomly assign destinations for demonstration purposes
            # In practice, you would likely set these based on your scenario
            x_max = self.env_params.additional_params['max_x']
            y_max = self.env_params.additional_params['max_y']
            for i, id in enumerate(ids):
                self.destinations[id] = [np.random.uniform(0, x_max), 
                                        np.random.uniform(0, y_max)]

        # Emission reward is MPG metric (does this need to be scaled by time?)
        distance_traveled = self.compute_distance_traveled(positions, self.prev_positions)
        
        # Avoid division by zero
        fuel_consumed = np.sum(np.array([self.k.vehicle.get_fuel_consumption(id) for id in ids]))
        emission_rewards = distance_traveled / max(fuel_consumed, 0.0001)

        # update prev positions
        self.prev_positions = positions

        # This will be a list of booleans where the ith element is true if positions[i] = destinations[i]
        destinations_reached = [(np.linalg.norm(positions[i] - self.destinations[ids[i]]) < 5.0) 
                              for i in range(len(ids))]
        
        # Agent accumulates -1 rewards for each vehicle that is not at its destination
        route_reward = np.sum(np.array([1 if reached else -1 for reached in destinations_reached]))

        return self.emission_weight * emission_rewards + self.route_weight * route_reward

    def get_state(self):
        ids = self.k.vehicle.get_rl_ids()
        speeds = [self.k.vehicle.get_speed(id) / self.k.network.max_speed() for id in ids]
        pos = [self.k.vehicle.get_2d_position(id) for id in ids]

        return np.array(speeds + pos)