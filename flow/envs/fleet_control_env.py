import re
import numpy as np
from flow.envs.base import Env
from random import randint

from gym.spaces import Tuple, Dict, MultiDiscrete, Discrete
from gym.spaces.box import Box
from flow.networks.fleet_grid import ADDITIONAL_NET_PARAMS

ADDITIONAL_ENV_PARAMS = {
    # Number of vehicles in the fleet
    'num_vehicles': 14,
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

        self.returnReward = 0
        self.tot_steps = 0
        self.total_mpg = 0
        
        self.prev_distances = np.zeros(env_params.additional_params["num_vehicles"])

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
                
                if not ((rand_x, rand_y) in self.destinations):
                    self.destinations.append((rand_x, rand_y))
                    same_dest = False

        # Initialize previous positions
        self.prev_positions = {}
        
        # Weights for different components of the reward function
        self.emission_weight = 1.0
        self.route_weight = 1.0
        self.all_vechicles = self.k.vehicle.get_ids()
        # print("all vechicles", self.all_vechicles)

        # print("Vehicle routes", self.k.vehicle.get_route(self.k.vehicle.get_rl_ids()))
    
    @property
    def action_space(self):
        accel_dim = self.initial_vehicles.num_rl_vehicles
        route_dim = self.initial_vehicles.num_rl_vehicles

        # Each action is formatted a sa flattened list which is required for compatibility
        # with Ray RLlib. The actions is a concatenated array with the following format:
        #       [accelerations for each vehicle] + [routing decisions for each vehicle]
        # return Box(
        #     low=np.array([-self.env_params.additional_params['max_decel']] * accel_dim +
        #                 [0] * route_dim), # Array combining lowest acceleration values + min indexed routing action
        #     high=np.array([self.env_params.additional_params['max_accel']] * accel_dim +
        #                 [3] * route_dim),  # Array combining largest acceleration values + max indexed routing action, assuming 4 possible routes (0-3)
        #     dtype=np.float32
        # )
        return Box(
            low=np.array([-self.env_params.additional_params['max_decel']] * accel_dim), # Array combining lowest acceleration values + min indexed routing action
            high=np.array([self.env_params.additional_params['max_accel']] * accel_dim),  # Array combining largest acceleration values + max indexed routing action, assuming 4 possible routes (0-3)
            dtype=np.float32
        )

    @property
    def observation_space(self):
        num_vehicles = self.initial_vehicles.num_rl_vehicles

        # NOTE: consider changing the observation from including (x,y) to including the edge

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
            # print("Veh id", veh_id, "Current edge:", current_edge)

            veh_pos = self.k.vehicle.get_2d_position(veh_id)
            
            near_intersection = False
            allowed_actions = []
            
            # Weird case that is not understood, don't update the route
            match = re.match(r"(bot|right|top|left)(\d+)_(\d+)", current_edge)
            if match is None:
                # print(match, "match is none", current_edge, "veh id", veh_id, "current route", current_route)
                routes.append(current_route)
                continue

            # Group(1) = {bot, right, top, left}
            # Group(2) = row number in network
            # Group(3) = col number in network
            row, col = int(match.group(2)), int(match.group(3))
            direction = match.group(1)
            
            

            intersections = self.network.get_inner_nodes()
            # calculate distance from vehicle to each intersection
            min_distance = 10000
            intersection = 5
            for node in intersections:
                
                distance = np.sqrt(
                    (veh_pos[0] - node["x"])**2 + 
                    (veh_pos[1] - node["y"])**2
                )
                if distance < min_distance:
                    min_distance = distance
                    intersection = node

                    # print("max distance", max_distance)
                # near intersection radius means the vehicle is near the intersection
                # print(node['radius'], "node radius")
                # TODO: find optimal distance

            if min_distance < 10:
                col_num = 2
                match = re.search(r'\d+', intersection['id'])  # Find one or more digits in the string
                node_num =  int(match.group()) if match else None  # Convert to int if found
                j = node_num % col_num
                i = node_num // col_num  
                # print("node id", intersection['id'], "i", i, "j", j)
                if current_edge == "top{}_{}".format(i, j) or current_edge == "bot{}_{}".format(i, j+1) or \
                    current_edge == "left{}_{}".format(i, j) or current_edge == "right{}_{}".format(i+1, j):
                    near_intersection = False
                else:

                    # print(" turn", veh_id, "current edge", current_edge, intersection['id'], "distance", min_distance)
                    near_intersection = True
                    allowed_actions = [0, 1, 2, 3]  
                    

            # Action decided by the RL agent
            action = int(abs((route_actions[i]))) % 4            
            
            
            
            
            # Calculate new route based on action if near intersection
            if near_intersection:
                # current_edge looks like bot{row}_{col}, or right{row}_{col}
                #the numbers have to be the index of the current edge , current_edge looks like bot0_1
                #############################################################################################
                # print("Veh id", veh_id, "near intersection","current edge", current_edge, "current route:",current_route,  )
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
                    # print("Unknown edge type", current_edge)
                    pass
                
                
                match = re.match(r"(bot|right|top|left)(\d+)_(\d+)", new_edge)
                if match is None:
                    new_edge = current_edge
                    # print("panic match is none", new_edge, "veh id", veh_id, "current edge", current_edge, "direction", direction_map[action])
                else:  
                    row, col = int(match.group(2)), int(match.group(3))
                    if (row < 0 or row > 2 or \
                        col < 0 or col > 2):
                        # print("out of bounds", row, col)
                        new_edge = current_edge



                if current_edge != new_edge:
                    routes.append((current_edge, new_edge))
                else:
                    routes.append((current_edge))
                
            else:
                # print("not near intersection", veh_id, "current edge", current_edge)
                routes.append((current_edge))
        
        return routes
    
    def _apply_rl_actions(self, rl_actions):
        ids = self.k.vehicle.get_rl_ids()
        num_vehicles = len(ids)

       
        # print("Apply rl actions - Vehicle routes", self.k.vehicle.get_route(self.k.vehicle.get_rl_ids()))
        # Split the flattened actions back into acceleration and routing components
        # The first half of the array of actions holds the acceleration values, while 
        # the second half holds the routing values
        # accel_actions = rl_actions[:num_vehicles]
        # route_actions = rl_actions[num_vehicles:]

        # print("Applying route actions", rl_actions)

        # print("Velocities before accel:", self.k.vehicle.get_speed(ids))
        # print("RL actions:", rl_actions)
        # Update vehicle accelerations
        self.k.vehicle.apply_acceleration(ids, rl_actions)

        # Update vehicle routes based on route actions
        # print(f"Current routes: {self.k.vehicle.get_route(ids)}")
        # routes = self.get_updated_route(ids, route_actions)
        # print(f"Updated routes: {routes}")
        # self.k.vehicle.choose_routes(ids, routes)

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
    
    def compute_edge_distance(self, prev_edges, curr_edges, routes):
        prev_idxs = np.array([routes.index(prev_edge) for prev_edge in prev_edges])
        curr_idxs = np.array([routes.index(curr_edge) for curr_edge in curr_edges])

        hops = curr_idxs - prev_idxs

        pos_along_curr_edge = self.k.vehicle.get
        distances = hops * 30  # edge length
        # hops = []
        # for i in range(len(prev_idxs)):
        #     if prev_idxs[i] <= curr_idxs[i]:
        #         hops.append(curr_idxs[i] - prev_idxs[i])
        #     else:
        #         hops.append(len(prev_idxs) + )


    def compute_reward(self, rl_actions, **kwargs):
        num_vehicles = self.env_params.additional_params["num_vehicles"]

        ids = self.k.vehicle.get_rl_ids()
        # print(ids, len(ids))
        # Get current (x,y) positions for each vehicle
        positions = np.array([self.k.vehicle.get_2d_position(id) for id in ids])

        # Initialize previous positions if they don't exist
        if self.prev_positions is None:
            self.prev_positions = positions
            return 0  # Return 0 reward on first step
        

        curr_edges = self.k.vehicle.get_edge(ids)
        routes = self.k.vehicle.get_route(ids)

        curr_distances = np.zeros(num_vehicles)
        np.copyto(curr_distances, self.prev_distances)

        for id in ids:
            vehicle_num = int(id.split("_")[-1])
            curr_distances[vehicle_num] = self.k.vehicle.get_distance(id)

        distances = curr_distances - self.prev_distances
        # print("Prev dis:", self.prev_distances)
        # print("Curr dis:", curr_distances)
        # print("Vehicle Dis:", distances)

        self.prev_distances = curr_distances

        # distances = np.array(self.k.vehicle.get_distance(ids))
        # dist = compute_edge_distance(prev_edges, curr_edges, routes)

        # Emission reward is MPG metric (does this need to be scaled by time?)
        # distance_traveled = self.compute_distance_traveled(positions, self.prev_positions)
        
        # Avoid division by zero
        fuels = np.array([self.k.vehicle.get_fuel_consumption(id) for id in ids])
        fuel_consumed = np.sum(fuels)
        emission_reward = np.sum(distances) / max(fuel_consumed, 0.0001)

        # print("Vehicle Vel after accel:", self.k.vehicle.get_speed(ids))
        # print("Vehicle Fue:", fuels)
        # print("Reward:", emission_reward)

        # update prev positions
        self.prev_positions = positions

        # This will be a list of booleans where the ith element is true if positions[i] = destinations[i]
        # destinations_reached = []
        # for i in range(len(ids)):
        #     dist = np.linalg.norm(np.array(positions[i]) - np.array(self.destinations[i]))
        #     if dist < 5.0:
        #         destinations_reached.append(True)
        #     else:
        #         destinations_reached.append(False)

        
        # Agent accumulates -1 rewards for each vehicle that is not at its destination
        # route_reward = np.sum(np.array([1 if reached else -1 for reached in destinations_reached]))

        self.tot_steps += 1
        self.total_mpg += emission_reward
        curr_reward = self.total_mpg/ self.tot_steps
        self.returnReward += curr_reward

        # print("Current Reward:",curr_reward)
        # print ("Total Reward:", self.returnReward )
        return emission_reward
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