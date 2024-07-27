
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

from gym_connect.utils.lattice import distance
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle


def render_thesis(self):
    if self.fig is None:
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))  # Ensure consistent DPI setting
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)

        self.points = self.ax.scatter(self.x[2:, 0], self.x[2:, 1], c=self.battery, cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1), edgecolor='black', s=210, zorder=2)
        self.points2, = self.ax.plot(self.x[0:2, 0], self.x[0:2, 1], '^', mfc='red', mec='black', markersize=15, zorder=2)

        self.lines = []
        self.viz_paths = None
        self.red_circles = []
        self.green_circles = []

        if self.mode != 'keyboard' and self.mode_change:
            self.viz_paths, = self.ax.plot(self.path_xy[:, 0], self.path_xy[:, 1], 'k--', zorder=1)
            self.mode_change = False

        plt.xlim(self.plot_lim[0], self.plot_lim[1])
        plt.ylim(self.plot_lim[2], self.plot_lim[3])
        self.ax.set_aspect('equal', 'box')

    else:
        for _, _, line in self.lines:
            line.remove()
        self.lines.clear()

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.x[i, :2] - self.x[j, :2]) < self.comm_radius:
                    line, = self.ax.plot([self.x[i, 0], self.x[j, 0]], [self.x[i, 1], self.x[j, 1]], 'k-', zorder=1, alpha=0.3, linewidth=2)
                    self.lines.append((i, j, line))

        if self.mode != 'keyboard' and self.mode_change:
            self.viz_paths.remove()
            self.viz_paths, = self.ax.plot(self.path_xy[:, 0], self.path_xy[:, 1], 'k--', zorder=1)
            self.mode_change = False

    for line_data in self.lines:
        line_data[2].set_data([self.x[line_data[0], 0], self.x[line_data[1], 0]], 
                                [self.x[line_data[0], 1], self.x[line_data[1], 1]])

    if self.mode != 'keyboard':
        self.viz_paths.set_xdata(self.path_xy[:, 0])
        self.viz_paths.set_ydata(self.path_xy[:, 1])

    self.points.set_offsets(self.x[2:, :2])
    self.points.set_array(self.battery)

    self.points2.set_xdata(self.x[0:2, 0])
    self.points2.set_ydata(self.x[0:2, 1])

    # Remove old red and green circles
    while self.red_circles:
        circle = self.red_circles.pop()
        circle.remove()
        
    # while self.green_circles:
    #     circle = self.green_circles.pop()
    #     circle.remove()

    # Highlight the agent with the lowest battery
    min_battery_idx = np.argmin(self.battery)
    min_battery_pos = self.x[min_battery_idx + 2, :2]  # Adjust index for points starting at 2

    # Draw a red dashed circle around the agent with the lowest battery
    red_circle = Circle(min_battery_pos, 0.2, edgecolor='red', facecolor='none', linestyle='--', linewidth=2, zorder=3)
    self.ax.add_patch(red_circle)
    self.red_circles.append(red_circle)
    
    # Highlight agents with battery greater than 0.98
    # high_battery_indices = np.where(self.battery > 0.98)[0]
    # for idx in high_battery_indices:
    #     high_battery_pos = self.x[idx + 2, :2]  # Adjust index for points starting at 2
    #     green_circle = Circle(high_battery_pos, 0.2, edgecolor='green', facecolor='none', linestyle='--', linewidth=2, zorder=3)
    #     self.ax.add_patch(green_circle)
    #     self.green_circles.append(green_circle)

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    # Get the current canvas dimensions
    width, height = self.fig.canvas.get_width_height()

    # Debugging: print dimensions and array size
    rgb_array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
    # print(f"Canvas dimensions: width={width}, height={height}")
    # print(f"Array size: {rgb_array.size}")
    # print(f"Expected size: {width * height * 3}")

    # Ensure the array size matches the expected size
    if rgb_array.size != width * height * 3:
        # Recalculate expected size to handle minor discrepancies
        dpi = self.fig.get_dpi()
        width_inch, height_inch = self.fig.get_size_inches()
        expected_width = int(width_inch * dpi)
        expected_height = int(height_inch * dpi)
        expected_size = expected_width * expected_height * 3
        # print(f"Recalculated expected size: {expected_size}")

        # Check if recalculated size matches the array size
        if rgb_array.size != expected_size:
            raise ValueError(f"Size mismatch: array size is {rgb_array.size}, but recalculated expected size is {expected_size}")

        # Update width and height based on recalculated values
        width, height = expected_width, expected_height

    # Return the RGB array for saving the video
    return rgb_array.reshape(height, width, 3)


def render_sim(self):
    """
    Render the environment with agents as points in 2D space
    """

    if self.fig is None:
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        
        self.points = self.ax.scatter(self.x[2:, 0], self.x[2:, 1], c=self.battery, cmap='Blues', norm=Normalize(vmin=0, vmax=1), edgecolor='black', s=100, zorder=2)
        
        self.points2, = self.ax.plot(self.x[0:2, 0], self.x[0:2, 1], '^', mfc='red', mec='black', markersize=10, zorder=2)
        self.ax.plot([0], [0], 'kx',zorder=1)

        # # Plotting the curve as a dotted line
        # if self.path_xy is not None:
            # curve_x, curve_y = self.path_xy[:,0], self.path_xy[:,1]  # Assume doesn't need 't' passed anymore
            # self.ax.plot(curve_x, curve_y, 'k--', zorder=1)  # 'k--' for black dotted line
        
        self.lines = []
        self.viz_paths = None

        if self.mode!='keyboard' and self.mode_change:
            self.viz_paths, = self.ax.plot(self.path_xy[:,0], self.path_xy[:,1], 'k--', zorder=1)
            self.mode_change = False
        
        plt.xlim(self.plot_lim[0], self.plot_lim[1])
        plt.ylim(self.plot_lim[2], self.plot_lim[3])
        a = plt.gca()
        a.set_xticks(a.get_xticks())
        a.set_xticklabels(a.get_xticks())
        a.set_yticks(a.get_yticks())
        a.set_yticklabels(a.get_yticks())
        a.grid()
        self.ax.set_aspect('equal', 'box')
        plt.title('Battery Aware Controller')
    else:
        # Remove old lines from the plot
        for i, j, line in self.lines:
            line.remove()
        self.lines.clear()

        # Draw new lines according to the current adjacency matrix
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):  # This ensures that you don't repeat lines
                if np.linalg.norm(self.x[i,:2] - self.x[j,:2]) < self.comm_radius:
                # if self.state_network[i, j] > 0:
                    line, = self.ax.plot([self.x[i, 0], self.x[j, 0]], [self.x[i, 1], self.x[j, 1]], 'k-',zorder=1, alpha=0.5)
                    self.lines.append((i, j, line))

        if self.mode!='keyboard' and self.mode_change:
            #clear old lines
            self.viz_paths.remove()
            self.viz_paths, = self.ax.plot(self.path_xy[:,0], self.path_xy[:,1], 'k--', zorder=1)
            self.mode_change = False

    # Assuming `self.lines` is a list that stores tuples of (i, j, line object)
    for line_data in self.lines:
        # Update line data here based on the current state_network and positions
        line_data[2].set_data([self.x[line_data[0], 0], self.x[line_data[1], 0]], 
                            [self.x[line_data[0], 1], self.x[line_data[1], 1]])

    if self.mode!='keyboard':
        self.viz_paths.set_xdata(self.path_xy[:,0])
        self.viz_paths.set_ydata(self.path_xy[:,1])

    # Update points
    # self.points.set_xdata(self.x[2:, 0])
    # self.points.set_ydata(self.x[2:, 1])

    # Update the offsets of the scatter plot directly
    self.points.set_offsets(self.x[2:, :2])
    self.points.set_array(self.battery) 

    self.points2.set_xdata(self.x[0:2, 0])
    self.points2.set_ydata(self.x[0:2, 1])

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()


def render_sim_3d(self):
    """
    Render the environment with agents as points in 3D space
    """
    if self.fig is None:
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        

        self.points = self.ax.scatter(self.x[2:, 0], self.x[2:, 1], self.x[2:, 2], c=self.battery, cmap='Blues', norm=Normalize(vmin=0, vmax=1), edgecolor='black', s=100, zorder=2)
        self.points2 = self.ax.scatter( self.x[0:2, 0], 
                                        self.x[0:2, 1], 
                                        self.x[0:2, 2], 
                                        marker='^', 
                                        c=[0.5,0.5],
                                        cmap='Reds',
                                        norm=Normalize(vmin=0, vmax=1),
                                        # c='red',  # This sets the face color
                                        edgecolors='black',  # This sets the edge color
                                        s=100,  # Adjust size as needed
                                        zorder=2)   
        
        # Add the base station or other static points
        self.ax.scatter([0], [0], [0], c='k', marker='x', zorder=1)

        # Configure axes limits
        self.ax.set_xlim(self.plot_lim[0], self.plot_lim[1])
        self.ax.set_ylim(self.plot_lim[2], self.plot_lim[3])
        self.ax.set_zlim(self.plot_lim[4], self.plot_lim[5])  # New z-axis limits
        
        # Configure labels and ticks
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid()
        plt.title('Battery Aware Controller in 3D')

        self.lines = []
        self.viz_paths = None

    else:
        # Update points for dynamic objects
        self.points._offsets3d = (self.x[2:, 0], self.x[2:, 1], self.x[2:, 2])
        self.points.set_array(self.battery) 

        self.points2._offsets3d = (self.x[:2, 0], self.x[:2, 1], self.x[:2, 2])
        self.points2.set_array([0.5,0.5]) 

        # Update lines if needed
        # Assuming you have logic to update or redraw lines based on the state network
        for i, j, line in self.lines:
            line.remove()
        self.lines.clear()

        # Draw new lines according to the current adjacency matrix
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):  # This ensures that you don't repeat lines
                if np.linalg.norm(self.x[i,:2] - self.x[j,:2]) < self.comm_radius:
                # if self.state_network[i, j] > 0:
                    line, = self.ax.plot([self.x[i, 0], self.x[j, 0]], [self.x[i, 1], self.x[j, 1]], [self.x[i, 2], self.x[j, 2]], 'k-',zorder=1, alpha=0.5)
                    self.lines.append((i, j, line))

    # Assuming `self.lines` is a list that stores tuples of (i, j, line object)
    for line_data in self.lines:
        # Update line data here based on the current state_network and positions
        line_data[2].set_data([self.x[line_data[0], 0], self.x[line_data[1], 0]], 
                            [self.x[line_data[0], 1], self.x[line_data[1], 1]])

    # Update the offsets of the scatter plot directly
    # self.points.set_offsets(self.x[2:, :2])
    # self.points.set_array(self.battery) 

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()
