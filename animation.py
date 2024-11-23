import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from simulator import Spaceship
from tqdm import tqdm

# for some Windows machines
# import matplotlib as mpl
# mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ProgramFiles2\\ffmpeg\\bin\\ffmpeg.exe'

class SimAnimation():
    def __init__(self, bodies, objective, limits, frame, iterations, i_episode, save_freq, title = "", draw_neighbourhood=False, grid_radius=None, box_width=None):
        print("animating")
        self.draw_neighbourhood = draw_neighbourhood
        if self.draw_neighbourhood and (grid_radius==None or box_width==None):
            raise Exception("If draw_neighbourhood is set to True, grid_radius and box_width must be specified") 
        self.grid_radius = grid_radius
        self.box_width = box_width
        self.bodies = bodies
        self.objective = objective
        self.states = frame
        self.limits = limits
        self.save_stride = save_freq
        fig, self.ax = plt.subplots(1, 1, figsize = (6, 6))
        anim = animation.FuncAnimation(fig, self.animate, frames = iterations, interval = 1)
        # plt.show()

        # works on windows too but needs to include path in imports above
        # saving to m4 using ffmpeg writer
        # title: `episode`_sim-name.mp4
        if (i_episode) % self.save_stride == 0:
            # writervideo = animation.FFMpegWriter(fps=60)
            # name = str(i_episode) + "_" + title + ".gif"
            # anim.save(name, writer=writervideo)
            # below is tested on nersc. works faster.
            writergif = animation.PillowWriter(fps=30)
            name = str(i_episode) + "_" + title + ".gif"
            with tqdm(total=iterations, desc="Saving animation", unit="frame") as pbar:
                anim.save(name, writer=writergif, progress_callback=lambda i, n: pbar.update(1))




    def animate(self, end):
        start = max(0, end-100)
        self.ax.cla()
        self.ax.scatter(self.objective[0], self.objective[1], marker='*', s=150)
        for body in self.bodies:
            self.ax.plot(body.history[start:end, 0], body.history[start:end, 1], ".", color=body.color)
            if isinstance(body, Spaceship) and self.draw_neighbourhood:
                self.draw_state(self.states[end], body.history[end])
        self.ax.set_xlim(-self.limits, self.limits)
        self.ax.set_ylim(-self.limits, self.limits)


    def draw_state(self, state, position):
        old_position = position
        position = position - np.ones(2) * (self.grid_radius + 0.5) * self.box_width
        for i in range(0, 2*self.grid_radius+1):
            for j in range(0, 2*self.grid_radius+1):
                if state[0, i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((position[0] + j*self.box_width, position[1] + i*self.box_width),
                                                    self.box_width, self.box_width, fill=True))
                elif state[1, i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((position[0] + j*self.box_width, position[1] + i*self.box_width),
                                                    self.box_width, self.box_width, fill=False, color="r", linewidth=2))
                else: self.ax.add_patch(plt.Rectangle((position[0] + j*self.box_width, position[1] + i*self.box_width),
                                                self.box_width, self.box_width, fill=False))