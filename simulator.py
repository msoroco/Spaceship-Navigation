import statistics
import numpy as np
import os
import json
import random
from collections import deque

G = 6.6

class Simulator:
    def __init__(self, filepath : str):
        """
        Create a simulation from a JSON

        Parameters
        ----------
        filepath : str
            Path to json file

        JSON file attributes
        ---------
        * `limits`: 1/2 width/height of the whole square environment. Defaults to 300.
        * `grid_radius`: The width & height in coordinates of the square frame around agent.
        * `box_width`: The width & height of a unit coordinate in the vector space.
        * `frames`: The number of past frames agent will maintain in addition to its observed frame.
        * `frame_stride`: The number of time steps between the frames the agent maintains.
        * `tolerance`: The distance (in coordinates) to objective within which agent must achieve.
          Also the distance within which agent is considered to have collided with another body.
        * `agent`:
        * `random_agent_position`: wether to set the agent position to be random on each start (will overwrite given JSON position)
        * `objective`: length 2 array of coordinate of the objective
        * `bodies`:
        * `start_zeros`: Have the simulation pad the missing frames with all zeros for first |`frames`| frames (default is do nothing)
        * `start_copies`: Have the simulation pad the missing frames with itself for first |`frames`| frames (`start_zeros` will default if both `True`)
        *`introduce_new_bodies`: Have the simulation introduce new bodies at random steps with `introduce_new_bodies`% probability
        """
        json_obj = Simulator.__load_json(filepath)
        self._json_obj = json_obj
        # State information
        self.reward_scheme = [0, self._json_obj["penalty"], self._json_obj["reward"]]
        self.limits = json_obj.get("limits", 500)
        self.grid_radius = json_obj.get("grid_radius", 10)
        self.box_width = json_obj.get("box_width", 20)
        self.frame_stride = json_obj.get("frame_stride", 1)
        self.frames = json_obj.get("frames", 4)
        self.penalty = self._json_obj.get("penalty", -10)
        self.tolerance = self._json_obj.get("tolerance", self.box_width*5)
        self.start_zeros = self._json_obj.get("start_zeros", True)
        self.start_copies = self._json_obj.get("self.start_copies", False)
        self.verbose = self._json_obj.get("verbose", False)
        self.random_agent_position = self._json_obj.get("random_agent_position", True)
        self.introduce_new_bodies = self._json_obj.get("introduce_new_bodies", 0)

    def get_bodies_and_objective(self):
        return self.bodies, self.objective
    
    def get_environment_info(self):
        """
        Returns the unit length mapping (box_width) and the width of environment.
        * limits`: 1/2 width/height of the whole square environment.
        * `box_width`: The width & height of a unit coordinate in the vector space.
        """
        return self.box_width, self.limits


    def info(self):
        """
        Returns a tuple (S, N) where S is the shape of the state
        and N is the number of actions (including do nothing)
        """
        n_actions = len(self.agent.actions) + 1

        return self.__current_state_shape, n_actions
    

    def start(self, seed=None):
        """
        Initializes all simulation elements to starting positions

        Returns state
        """

        if seed is not None:
            random.seed(seed)
        # TODO: change this to use rng if NONE (for velocity)
        if self.random_agent_position:
            agent_position = np.random.uniform(-self.limits, self.limits, size=2)
            self._json_obj["agent"]["position"] = agent_position

        self.agent = Spaceship(** self._json_obj["agent"])

        self.bodies = []
        # self.bodiesQueues = [] # to track the evolution of bodies for heatmap
        bodies_list = self._json_obj["bodies"]
        for body in bodies_list:
            self.bodies.append(Body(**body))
            # self.bodiesQueues.append(deque([], maxlen=self.frames*self.frame_stride))
        self.bodies.insert(0, self.agent)
        
        try: self.objective = np.array(self._json_obj["objective"])
        except: self.objective = np.random.uniform(-self.limits, self.limits, size=2)

        # Empty past frame queue
        self.past_frames = deque([], maxlen=self.frames*self.frame_stride)
        return self.__get_state()
    

    def step(self, action : int):
        """
        Continues the simulation by one step with the given action

        Returns the next state and reward and termination condition
        * `termination_condition` is:
                * 0 if no termination reward
                * 1 if a penalty is applied
                * 2 if a reward is applied for reaching the objective
        """
        sample = random.uniform(0, 1)
        if sample < self.introduce_new_bodies:
            self.__add_new_body()


        self.agent.do_action(action)
        for body in self.bodies:
            body.step()
        for body1 in self.bodies:
            for body2 in self.bodies:
                if body1 != body2:
                    body1.gravity(body2)

        state = self.__get_state()
        termination_condition = self.__get_terminated()
        reward = self.__get_reward(termination_condition)
        return state, reward, termination_condition

    def __get_terminated(self):
        """
        Checkes whether the simulation should terminate if the spaceship has reached the objective,
        crashed into (or went through) a planet, or went outside the simulation zone.

        Returns a termination condition, 0 means not terminated, 1 if a penalty is applied, and 2 if
        a reward for reach the objective

        Returns termination_condition.
            * `termination_condition` is:
                * 0 if no termination reward
                * 1 if a penalty is applied
                * 2 if a reward is applied for reaching the objective
        """
        termination_condition = 0
        # Objective reached
        if np.linalg.norm(self.objective - self.agent.position) < self.tolerance: 
            terminated = True
            termination_condition = 2
            if self.verbose: print("Termination: Objective reached!")
        # Out of bounds
        elif abs(self.agent.position[0]) > self.limits or abs(self.agent.position[1]) > self.limits:
            termination_condition = 1
            # if self.verbose: print(f"Termination: out of bounds at position {self.agent.position}")
        # Check crash or Agent going through a body:
        else:
            for body in self.bodies:
                if body != self.agent:
                    # Crash
                    if np.linalg.norm(body.position - self.agent.position) < self.tolerance:
                        termination_condition = 1
                        if self.verbose: print("Termination: agent collided with a body")
                    # Agent went through a body:
                    curr_pos = self.agent.position
                    prev_pos = self.agent.history[-2]
                    agent_line = (curr_pos - prev_pos)
                    body_pos = (body.position - prev_pos)
                    projection_body = np.dot(agent_line, body_pos)/(np.dot(agent_line, agent_line)+1e6 )* agent_line
                    # check that distance between orthogonal projection of body into agents path and body is less than tolerance
                    if np.linalg.norm(body.position - (projection_body + prev_pos)) < self.tolerance:
                        # check that body lies between the two positions of agent
                        if max(curr_pos[0], prev_pos[0]) >= (projection_body + prev_pos)[0] and min(curr_pos[0], prev_pos[0]) <= (projection_body + prev_pos)[0]:
                            if max(curr_pos[1], prev_pos[1]) >= (projection_body + prev_pos)[1] and min(curr_pos[1], prev_pos[1]) <= (projection_body + prev_pos)[1]:
                                if self.verbose: print("Termination: agent went through a body")
                                termination_condition = 1
        return termination_condition

    def __get_reward(self, reward_index):
        reward = self.reward_scheme[reward_index]  # Base reward based on termination condition
        
        # Penalize for nearing boundaries
        if abs(self.agent.position[0]) > self.limits - self.tolerance or abs(self.agent.position[1]) > self.limits - self.tolerance:
            reward -= 2  # Reduced penalty for being near boundaries

        # Reward for staying within bounds
        if abs(self.agent.position[0]) < self.limits and abs(self.agent.position[1]) < self.limits:
            reward += 0.1  # Small reward for staying within bounds

        # Penalize for collisions with bodies
        for body in self.bodies:
            if body != self.agent:
                distance_to_body = np.linalg.norm(self.agent.position - body.position)
                if distance_to_body < self.tolerance:
                    reward -= 5  # Moderate penalty for collision

                # Penalize for getting too close to a body
                if distance_to_body < self.tolerance * 2:
                    reward -= 3  # Light penalty for approaching a body

                # Reward for staying away from bodies
                if distance_to_body > self.tolerance:
                    reward += 0.2  # Reward for keeping distance from bodies

        # Add reward for reaching the objective
        if np.linalg.norm(self.agent.position - self.objective) < self.tolerance:
            reward += 10  # Reward for reaching the objective

        # Additional scaling based on distance to the objective
        reward -= 0.01 * np.clip(1 - self.tolerance / np.linalg.norm(self.objective - self.agent.position), 0, 1)

        return reward



    def __get_state(self, given_position=None, forHeatmap=False):
        if forHeatmap:
            frame = self.__get_current_frame(given_position)
        else:
            frame = self.__get_current_frame()
        # Create state
        state = frame
        # Attach past frames
        for i in range(self.frames):
            if forHeatmap: # for heatmap
                state = np.concatenate((state, np.zeros(frame.shape)))
            elif len(self.past_frames) >= (i+1)*self.frame_stride:
                state = np.concatenate((state, self.past_frames[-(i+1)*self.frame_stride]))
            elif self.start_zeros: # TODO: If you can't attach a past frame, attach a dummy frame
                state = np.concatenate((state, np.zeros(frame.shape)))
            elif self.start_copies: # If you can't attach a past frame, attach a copy of itself
                state = np.concatenate((state, frame))
        if forHeatmap: # do not update state
            return state
        # Update info
        self.past_frames.append(frame) # deque will automatically evict oldest frame if full
        self.__current_state_shape = state.shape
        return state
    
    def __add_new_body(self):
        edge = random.choice([0, 1, 2, 3])
        position = np.random.uniform(-self.limits, self.limits)
        if edge == 0:  # Top edge
            coordinate = (position, self.limits)
        elif edge == 1:  # Right edge
            coordinate = (self.limits, position)
        elif edge == 2:  # Bottom edge
            coordinate = (position, -self.limits)
        elif edge == 3:  # Left edge
            coordinate = (-self.limits, position)

        colors = ["cyan", "purple"]

        # Randomly choose a color
        color = random.choice(colors)

        body = Body(mass=0, position=coordinate, velocity=np.random.uniform(0, 0.4, size=2), color=color)
        self.bodies.append(body)
        return


    def __get_current_frame(self, given_position=None):
        if given_position is None:
            given_position = self.agent.position

        radius = (self.grid_radius + 0.5) * self.box_width
        obstacle_grid = np.zeros((2*self.grid_radius+1, 2*self.grid_radius+1))
        objective_grid = np.zeros((2*self.grid_radius+1, 2*self.grid_radius+1))

        # Assign objective
        # Transform and shift to bottom left corner of grid
        position = self.objective - given_position
        # TODO: rotation goes here
        position = position + radius*np.ones(2)
        index = np.clip(np.floor(position/self.box_width).astype(int), 0, 2*self.grid_radius)
        objective_grid[index[1], index[0]] = 1

        # Assign obstacles
        for body in self.bodies:
            if body != self.agent:
                position = body.position - given_position
                # TODO: rotation goes here
                if np.max(np.abs(position)) <= radius:
                    position = position + radius*np.ones(2)
                    index = np.clip(np.floor(position/self.box_width).astype(int), 0, 2*self.grid_radius)
                    obstacle_grid[index[1], index[0]] = 1

        # Create frame       
        frame = np.stack((obstacle_grid, objective_grid), axis=0)
        return frame
    
    # for animation. May not be needed
    def get_current_frame(self):
        return self.__get_current_frame()
    
    
    def get_current_state(self, position):
        """ For heatmap. Gets the state around at given position (as if agent was there) (past frames are 0)
        """
        return self.__get_state(position, forHeatmap=True)

    def __load_json(filepath):
        with open(filepath) as json_file:
            json_obj = json.load(json_file)
        return json_obj


class Body:
    def __init__(self, mass, position, velocity, color, fixed=False):
        self.mass = float(mass)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.color = color
        self.history = np.array([position])
        self.fixed = fixed

    def step(self):
        if not self.fixed:
            self.velocity += 5*self.acceleration
            self.position += 5*self.velocity
            self.acceleration = np.zeros(2, dtype=float)
            self.history = np.vstack([self.history, self.position])
        else:
            self.history = np.vstack([self.history, self.position])

    # Compute gravity on us by a body
    def gravity(self, body):
        r = body.position - self.position
        self.acceleration += r * (G * body.mass / np.linalg.norm(r)**3)  # a_s = F/m_s = G * (m_b * m_s)/r*2 *(1 / m_s)


class Spaceship(Body):
    def __init__(self, position, velocity, color, speed):
        super().__init__(0, position, velocity, color)
        self.actions = [self.thrust_up, self.thrust_down, self.thrust_left, self.thrust_right]
        self.speed = speed
        self.speed *= 0.5  # Reduce movement speed

    
    # Accepts int for each of 4 possible actions
    def do_action(self, id):
        if id < len(self.actions):
            self.actions[id]()
        # If given id greater than 4, do nothing

    def thrust_up(self):
        self.position += np.array([0, self.speed], dtype=float)
    
    def thrust_down(self):
        self.position += np.array([0, -self.speed], dtype=float)

    def thrust_left(self):
        self.position += np.array([-self.speed, 0], dtype=float)

    def thrust_right(self):
        self.position += np.array([self.speed, 0], dtype=float)