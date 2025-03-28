import re
import numpy as np
from flow.envs.base import Env
from random import randint

from gym.spaces import Tuple, Dict, MultiDiscrete, Discrete
from gym.spaces.box import Box
from flow.networks.fleet_grid import ADDITIONAL_NET_PARAMS

ADDITIONAL_ENV_PARAMS = {
    # Number of vehicles in the fleet
    'num_vehicles': 4,
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
        self.destinations = []
        edges = network.specify_edges(None)
        for i in range(env_params.additional_params["num_vehicles"]):
            same_dest = True

            # Keep generating a random position until you make a new one
            while same_dest:
                # Obtain a random edge in the network + a random position on that edge
                rand_edge = edges[randint(0, len(edges)-1)]
                rand_pos = randint(0, rand_edge["length"])

                from_node = next(node for node in network.specify_nodes(None) if node['id'] == rand_edge['from'])
                to_node = next(node for node in network.specify_nodes(None) if node['id'] == rand_edge['to'])

                x_start, y_start = from_node['x'], from_node['y']
                x_end, y_end = to_node['x'], to_node['y']

                # Calculate direction vector
                dx = abs(x_end - x_start)
                dy = abs(y_end - y_start)

                # Normalize direction vector
                length = ((dx ** 2) + (dy ** 2)) ** 0.5
                dx /= length
                dy /= length

                # Compute (x, y) at the given position
                rand_x = x_start + dx * rand_pos
                rand_y = y_start + dy * rand_pos
                
                # print("Rand edge", rand_edge)
                # print("Rand pos", rand_pos)
                # print("x_start", x_start)
                # print("y_start", y_start)
                # print("dx", dx)
                # print("dy", dy)
                # print(f"Vehicle {i} candidate dest {rand_x, rand_y}")
                if not ((rand_x, rand_y) in self.destinations):
                    # print(f"Vehicle {i} received dest {rand_x, rand_y}")
                    self.destinations.append((rand_x, rand_y))
                    same_dest = False

        # Initialize previous positions
        self.prev_positions = {}
        
        # Weights for different components of the reward function
        self.emission_weight = 1.0
        self.route_weight = 1.0
        self.all_vechicles = self.k.vehicle.get_ids()
        # print("all vechicles", self.all_vechicles)
    
    @property
    def action_space(self):
        accel_dim = self.initial_vehicles.num_rl_vehicles
        route_dim = self.initial_vehicles.num_rl_vehicles

        # Each action is formatted a sa flattened list which is required for compatibility
        # with Ray RLlib. The actions is a concatenated array with the following format:
        #       [accelerations for each vehicle] + [routing decisions for each vehicle]
        return Box(
            low=np.array([-self.env_params.additional_params['max_decel']] * accel_dim +
                        [0] * route_dim), # Array combining lowest acceleration values + min indexed routing action
            high=np.array([self.env_params.additional_params['max_accel']] * accel_dim +
                        [3] * route_dim),  # Array combining largest acceleration values + max indexed routing action, assuming 4 possible routes (0-3)
            dtype=np.float32
    )

    @property
    def observation_space(self):
        num_vehicles = self.initial_vehicles.num_rl_vehicles

        # Each observation is formatted as a flattened list which is required for compatibility
        # with Ray RLlib. The observation is a concatenated array with the following format:
        #       [normalized velocities for each vehicle] + [x positions for each vehicle] + [y positions for each vehicle]
        return Box(
            low=np.array([0] * num_vehicles +  # Velocity lower bounds
                        [self.env_params.additional_params['min_x']] * num_vehicles +  # X position bounds
                        [self.env_params.additional_params['min_y']] * num_vehicles),  # Y position bounds
            high=np.array([1] * num_vehicles +  # Velocity upper bounds
                        [self.env_params.additional_params['max_x']] * num_vehicles +  # X position bounds
                        [self.env_params.additional_params['max_y']] * num_vehicles),  # Y position bounds
            dtype=np.float32
    )

    
    def get_updated_route(self, ids, route_actions):
        # print("in updated route")
        conns = self.network.specify_connections(None)

        # Map of actions to direction names for route calculations
        direction_map = {0: "top", 1: "bot", 2: "right",  3: "left"}
        # print("Connections", conns)

        # Stores each vehicle's final route
        routes = []
        
        for i, veh_id in enumerate(ids):
            # Get current edge and position of the vehicle
            # this function maybe wrong/ get_route
            current_edge = self.k.vehicle.get_edge(veh_id)
            current_route = self.k.vehicle.get_route(veh_id)
            print("Veh id", veh_id, "Current edge:", current_edge)

            veh_pos = self.k.vehicle.get_2d_position(veh_id)
            
            near_intersection = False
            allowed_actions = []
            
            
            
            

            intersections = self.network.get_inner_nodes()
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
                    

            # Action decided by the RL agent
            action = abs(int(route_actions[i]))            
            
            
            # Weird case that is not understood, don't update the route
            match = re.match(r"(bot|right|top|left)(\d+)_(\d+)", current_edge)
            if match is None:
                print(match, "match is none", current_edge, "veh id", veh_id)
                routes.append(current_route)
                continue

            # Group(1) = {bot, right, top, left}
            # Group(2) = row number in network
            # Group(3) = col number in network
            row, col = int(match.group(2)), int(match.group(3))
            direction = match.group(1)
            
            # Calculate new route based on action if near intersection
            if near_intersection:
                # current_edge looks like bot{row}_{col}, or right{row}_{col}
                #the numbers have to be the index of the current edge , current_edge looks like bot0_1
                #############################################################################################
                print("Veh id", veh_id, "near intersection")
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
                
                if current_edge != new_edge:
                    routes.append((current_edge, new_edge))
                
            else:
                routes.append((current_edge))
        
        return routes
    
    def _apply_rl_actions(self, rl_actions):
        ids = self.k.vehicle.get_rl_ids()
        num_vehicles = len(ids)

        print("Applying actions", rl_actions)

        # Split the flattened actions back into acceleration and routing components
        # The first half of the array of actions holds the acceleration values, while 
        # the second half holds the routing values
        accel_actions = rl_actions[:num_vehicles]
        route_actions = rl_actions[num_vehicles:]

        # Update vehicle accelerations
        self.k.vehicle.apply_acceleration(ids, accel_actions)

        # Update vehicle routes based on route actions
        print(f"Current routes: {self.k.vehicle.get_route(ids)}")
        routes = self.get_updated_route(ids, route_actions)
        print(f"Updated routes: {routes}")
        self.k.vehicle.choose_routes(ids, routes)

    def compute_distance_traveled(self, curr_positions, prev_positions):
        # Calculate element-wise Euclidean distances of each vehicle)
        # print("curr positions", curr_positions, "prev positions", prev_positions)
        distances = []
        if len(prev_positions) > 0 and len(curr_positions) >0:
            for i in range(len(curr_positions)):
                # Calculate distance for each vehicle
                dist = np.sqrt(np.power(curr_positions[i][0] - prev_positions[i][0], 2) + 
                               np.power(curr_positions[i][1] - prev_positions[i][1], 2))
                distances.append(dist)
            # distances = np.sqrt(np.power(curr_positions[0] - prev_positions[0], 2) + 
            #                 np.power(curr_positions[1] - prev_positions[1], 2))
        else:
            distances = np.zeros(len(curr_positions))
        # Return cumulative distance for the fleet
        return np.sum(distances)
    
    def compute_reward(self, rl_actions, **kwargs):
        ids = self.k.vehicle.get_rl_ids()
        # print(ids, len(ids))
        # Get current (x,y) positions for each vehicle
        positions = np.array([self.k.vehicle.get_2d_position(id) for id in ids])

        # Initialize previous positions if they don't exist
        if self.prev_positions is None:
            self.prev_positions = positions
            return 0  # Return 0 reward on first step
        
        # Emission reward is MPG metric (does this need to be scaled by time?)
        distance_traveled = self.compute_distance_traveled(positions, self.prev_positions)
        
        # Avoid division by zero
        fuel_consumed = np.sum(np.array([self.k.vehicle.get_fuel_consumption(id) for id in ids]))
        emission_rewards = distance_traveled / max(fuel_consumed, 0.0001)

        # update prev positions
        self.prev_positions = positions

        # This will be a list of booleans where the ith element is true if positions[i] = destinations[i]
        destinations_reached = []
        for i in range(len(ids)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(self.destinations[i]))
            if dist < 5.0:
                destinations_reached.append(True)
            else:
                destinations_reached.append(False)

        
        # Agent accumulates -1 rewards for each vehicle that is not at its destination
        route_reward = np.sum(np.array([1 if reached else -1 for reached in destinations_reached]))

        return self.emission_weight * emission_rewards + self.route_weight * route_reward

    def get_state(self):        
         
        num_vech  = ADDITIONAL_ENV_PARAMS["num_vehicles"]
        # print("in get state")

        ids = self.k.vehicle.get_rl_ids()
            
        # Collect all the speeds and (x,y) positions of the vehicles
        speeds = []
        pos = []
        for i in range(num_vech):
            id = 'rl_{}'.format(i)
            if id in ids:
                speeds.append(self.k.vehicle.get_speed(id) / self.k.network.max_speed())
                pos.append(self.k.vehicle.get_2d_position(id))
            else:
                speeds.append(0)
                # pos.append([-1, -1])
                pos.append([0, 0])
        # speeds = [self.k.vehicle.get_speed(id) / self.k.network.max_speed() for id in ids]
        # pos = [self.k.vehicle.get_2d_position(id) for id in ids]

        # Split positions into x values and y values to match format of states
        # for the RL agents
        # print("positions", pos)
        x_vals = [p[0] for p in pos]
        y_vals = [p[1] for p in pos]
        # out = np.array(speeds + x_vals + y_vals)
        # print("california", out, out.shape,)
        # print("speeds", speeds,pos)
        
        # save the first state in global variable 
        # self.state = out
        return np.array(speeds + x_vals + y_vals)

# python3 train.py flowagent --rl_trainer 'rllib' --num_cpus 4 --num_steps 5 --rollout_size 1