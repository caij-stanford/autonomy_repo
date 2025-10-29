#!/usr/bin/env python3

import typing as T
import math
import rclpy
import numpy as np
import scipy.interpolate  
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import yaw_to_quaternion
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D

V_PREV_THRES = 0.0001

# From P1 AStar. We just use methods from this class as in the P1 ipynb.

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1, norm_type=2):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.norm_type = norm_type # what norm to calculate (e.g. L1, L2, Linf)
        
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        # Always allow the true goal
        if x == self.x_goal:
            return True

        # Horizon bounds
        if (x[0] < self.statespace_lo[0] or x[0] > self.statespace_hi[0] or
            x[1] < self.statespace_lo[1] or x[1] > self.statespace_hi[1]):
            return False
        
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2), ord=self.norm_type)
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        d = self.resolution
        eight_direction = [(-d, -d), (-d, d), (d, d), (d, -d), # diagonals
                           (0, d), (0, -d), (d, 0), (-d, 0)] # cardinal
        
        for (dx, dy) in eight_direction:
            new_tuple = self.snap_to_grid((x[0] + dx, x[1] + dy)) 
            if self.is_free(new_tuple):
                neighbors.append(new_tuple)
                
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        def is_inside_horizon(x, lo, hi):
            return lo[0] <= x[0] <= hi[0] and lo[1] <= x[1] <= hi[1]
        
        # Clip the goal to a temporary goal as far as the horizon can reach:
        if not is_inside_horizon(self.x_goal, self.statespace_lo, self.statespace_hi):
            # Move temporary goal to the edge of the horizon in the direction of true goal
            x_goal_temp = np.clip(
                self.x_goal,
                self.statespace_lo,
                self.statespace_hi
            )
        else:
            x_goal_temp = self.x_goal
                
        # Open and closed sets are created in init, along w initial assignments
                
        while self.open_set:
            # Pop frontier into [x_curr]
            x_curr = self.find_best_est_cost_through()
            
            # At goal
            if x_curr == x_goal_temp:
                self.path = self.reconstruct_path()
                return True
            
            self.open_set.remove(x_curr)
            self.closed_set.add(x_curr)
            
            for x_neigh in self.get_neighbors(x_curr):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost = self.cost_to_arrive[x_curr] + self.distance(x_curr, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost > self.cost_to_arrive[x_neigh]: 
                    continue
                self.came_from[x_neigh] = x_curr
                self.cost_to_arrive[x_neigh] = tentative_cost
                self.est_cost_through[x_neigh] = tentative_cost + self.distance(x_neigh, x_goal_temp)
              
        # Failure if reached end        
        return False
        
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

#### Helpers #####

def stoch_to_det(stoch_grid: StochOccupancyGrid2D, thresh: float = 0.5) -> DetOccupancyGrid2D:
    """
    Convert a stochastic occupancy grid to a deterministic grid using a threshold,
    using vectorized operations for speed.

    Args:
        stoch_grid (StochOccupancyGrid2D): stochastic occupancy grid
        thresh (float): threshold probability for occupancy (default 0.5)

    Returns:
        DetOccupancyGrid2D
    """
    resolution = stoch_grid.resolution
    probs = stoch_grid.probs  # shape (height, width)

    # Create a boolean mask of occupied cells
    occupied_mask = probs >= thresh

    # Find the indices of occupied cells
    ys, xs = np.nonzero(occupied_mask)  # row indices = y, column indices = x

    # Convert grid indices to world coordinates
    obstacles = []
    for x_idx, y_idx in zip(xs, ys):
        x0, y0 = stoch_grid.grid2state(np.array([x_idx, y_idx]))
        x1, y1 = x0 + resolution, y0 + resolution
        obstacles.append(((x0, y0), (x1, y1)))

    det_grid = DetOccupancyGrid2D(
        width=stoch_grid.size_xy[0] * resolution,
        height=stoch_grid.size_xy[1] * resolution,
        obstacles=obstacles
    )
    return det_grid

### From P1 ipynb

def compute_smooth_plan(path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
    # Ensure path is a numpy array
    path = np.asarray(path)

    # Compute and set the following variables:
    #   1. ts: 
    #      Compute an array of time stamps for each planned waypoint assuming some constant 
    #      velocity between waypoints. 
    #
    #   2. path_x_spline, path_y_spline:
    #      Fit cubic splines to the x and y coordinates of the path separately
    #      with respect to the computed time stamp array.
    #      Hint: Use scipy.interpolate.splrep
    
    ##### YOUR CODE STARTS HERE #####
    diffs = np.diff(path, axis=0) # n-1 array of diff from t to t-1
    dists = np.linalg.norm(diffs, axis=1) # norm of diff is the dist between points
    dist_traveled = np.cumsum(dists)
    dist_traveled = np.insert(dist_traveled, 0, 0.0) # insert first element to remake length n
    
    ts = dist_traveled / v_desired
    path_x_spline = scipy.interpolate.splrep(ts, path[:, 0], s=spline_alpha, k=3)
    path_y_spline = scipy.interpolate.splrep(ts, path[:, 1], s=spline_alpha, k=3)
    ###### YOUR CODE END HERE ######
    
    return TrajectoryPlan(
        path=path,
        path_x_spline=path_x_spline,
        path_y_spline=path_y_spline,
        duration=ts[-1],
    )

class HeadingNavigator(BaseNavigator):
    def __init__(self,kpx: float = 1.0, kpy: float = 1.0, kdx: float = 0.2, kdy: float = 0.2,
                 V_max: float = 0.5, kp: float = 2.0):
        super().__init__("my_navigator")
        self.kp = kp
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.V_max = V_max
        # self.om_max is already defined as read only? Also not needed because we don't clip
        
        # Potential overrides for BaseNavigator's replan
        # self.replan_distance_threshold = 0.25  # meters, default ~0.25
        # self.replan_heading_threshold  = 0.2   # radians, default ~0.2
        # self.replan_time_threshold     = 4.0   # seconds between replans
        
        # # Potential overrides for BaseNavigator's alignment
        # self.align_threshold = 0.1  # radians (~6°)
        
        # Reset prev values
        self.V_prev = 0.
        self.t_prev = 0.
        self.om_prev = 0.
      
    def compute_heading_control(self, curr: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = wrap_angle(goal.theta - curr.theta)
        omega_control = self.kp * heading_error 
        return TurtleBotControl(omega=omega_control)
  
    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        
        x_d = scipy.interpolate.splev(t, plan.path_x_spline, der=0)
        xd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=1)
        xdd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=2)
        y_d = scipy.interpolate.splev(t, plan.path_y_spline, der=0)
        yd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=1)
        ydd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=2)
        
        dt = t - self.t_prev
        x = state.x 
        y = state.y 
        th = state.theta
        
        ########## Code starts here ##########
        # avoid singularity
        if abs(self.V_prev) < V_PREV_THRES:
            self.V_prev = V_PREV_THRES

        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                          [np.sin(th), self.V_prev*np.cos(th)]])
        a, om = np.linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########

        # # apply control limits
        # V = np.clip(V, -self.V_max, self.V_max)
        # om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        
        return TurtleBotControl(v = V, omega=om)
  
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, 
                occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> T.Optional[TrajectoryPlan]:
        
        # From Ed: horizon should limit the search to a square of that size around the state.
        # This is encoded into our algorithm as statespace lo and hi. 
        # Let the horizon be the SQUARE of (horizon) x (horizon) around (state).
        half = horizon / 2.0
        statespace_lo = (state.x - half, state.y - half)
        statespace_hi = (state.x + half, state.y + half)
        
        # We always search with our state at the center of the square
        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)
        
        # Run A* algorithm using norm 2 and above specs
        astar = AStar(statespace_lo, statespace_hi, x_init, x_goal, stoch_to_det(occupancy), resolution=resolution, norm_type=2)
        
        # Not valid path
        if not astar.solve() or len(astar.path) < 4:
            return None
        
        # Path found: reset class properties 
        self.reset()
        return compute_smooth_plan(astar.path) # Returns a TrajectoryPlan
        

if __name__ == "__main__":
    rclpy.init()
    node = HeadingNavigator()
    rclpy.spin(node)
    rclpy.shutdown()