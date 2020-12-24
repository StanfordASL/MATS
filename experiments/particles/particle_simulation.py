import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import dill
from tqdm import trange

# convention: metric units (m, kg, s)

class Agent:
    def __init__(self, position, velocity, mass=65, charge=5, radius=0.5, dt=0.001):
        self.x = position[0]
        self.y = position[1]
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.ax = None
        self.ay = None
        self.mass = mass
        self.charge = charge
        self.radius = radius
        self.net_force = None
        self.dt = dt

        self.x_history = [position[0]]
        self.y_history = [position[1]]
        self.vx_history = [velocity[0]]
        self.vy_history = [velocity[1]]
        self.ax_history = [0]
        self.ay_history = [0]


    def apply_net_force(self):
        # integrate dynamics
        self.ax = self.net_force[0] / self.mass
        self.ay = self.net_force[1] / self.mass

        self.vx = self.ax * self.dt + self.vx
        self.vy = self.ay * self.dt + self.vy

        self.x = self.vx * self.dt + self.x
        self.y = self.vy * self.dt + self.y

        # append to history
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.vx_history.append(self.vx)
        self.vy_history.append(self.vy)
        self.ax_history.append(self.ax)
        self.ay_history.append(self.ay)

        # reset force for next iteration
        self.ax = None
        self.ay = None
        self.net_force = None


    def add_force(self, force):
        if self.net_force is None:
            self.net_force = force
        else:
            self.net_force += force


    def history(self, dt=0.1):
        idx_spacing = int(dt // self.dt)

        return {('position', 'x'): np.array(self.x_history)[::idx_spacing],
                ('position', 'y'): np.array(self.y_history)[::idx_spacing],
                ('velocity', 'x'): np.array(self.vx_history)[::idx_spacing],
                ('velocity', 'y'): np.array(self.vy_history)[::idx_spacing],
                ('acceleration', 'x'): np.array(self.ax_history)[::idx_spacing],
                ('acceleration', 'y'): np.array(self.ay_history)[::idx_spacing]}


class Scenario:
    def __init__(self, num_agents, agent_mass=65, agent_radius=0.5, env_radius=10, init_speed_range=[4,12]):
        self.env_radius = env_radius
        self.agent_list = []

        self.initialize_agents(num_agents, agent_radius, env_radius, init_speed_range)


    def initialize_agents(self, num_agents, agent_radius, env_radius, init_speed_range):
        min_speed = init_speed_range[0]
        max_speed = init_speed_range[1]

        angle_range = np.linspace(0, 2*np.pi, 360, endpoint=False)
        for i in range(num_agents):
            # sample initial angle, speed
            angle = np.random.choice(angle_range)
            speed = (max_speed - min_speed) * np.random.random() + min_speed

            # create agent
            agent_pos = np.array([env_radius * np.cos(angle), env_radius * np.sin(angle)])
            agent_vel = [-speed * np.cos(angle), -speed * np.sin(angle)]
            self.agent_list.append(Agent(agent_pos, agent_vel, agent_radius))

            # eliminate initial angles that could result in initial collision for future agents
            dtheta = 2 * np.arcsin(agent_radius / (2 * env_radius))
            angle_range = angle_range[(angle_range > (angle + dtheta)) | (angle_range < (angle - dtheta))]


    def reset_other_agent(self, agent_radius=0.5, env_radius=10, init_speed_range=[4,12]):
        num_agents = len(self.agent_list)

        min_speed = init_speed_range[0]
        max_speed = init_speed_range[1]
        angle_range = np.linspace(0, 2*np.pi, 360, endpoint=False)
        for i in range(1, num_agents):
            agent = self.agent_list[i]

            # eliminate initial angles that could result in initial collision with robot
            dtheta = 2 * np.arcsin(agent_radius / (2 * env_radius))
            robot_angle = np.arctan2(self.agent_list[0].y_history[0], self.agent_list[0].x_history[0])
            angle_range = angle_range[(angle_range > (robot_angle + dtheta)) | (angle_range < (robot_angle - dtheta))]

            # sample initial angle, speed
            angle = np.random.choice(angle_range)
            speed = (max_speed - min_speed) * np.random.random() + min_speed

            # reset agent
            agent_pos = np.array([env_radius * np.cos(angle), env_radius * np.sin(angle)])
            agent_vel = [-speed * np.cos(angle), -speed * np.sin(angle)]
            self.agent_list[i] = Agent(agent_pos, agent_vel, agent_radius)

            # eliminate initial angles that could result in initial collision with particle
            angle_range = angle_range[(angle_range > (angle + dtheta)) | (angle_range < (angle - dtheta))]


    def simulate_iteration(self):
        # compute pairwise interactions (forces)
        for pair in list(combinations(self.agent_list, 2)):
            agent_a = pair[0]
            agent_b = pair[1]

            r = np.linalg.norm([agent_a.x - agent_b.x, agent_a.y - agent_b.y])
            theta = np.arctan2(agent_b.y - agent_a.y, agent_b.x - agent_a.x)
            F =  4 * agent_a.charge * agent_b.charge / r**2 # based on Coulomb force with modified Coulomb constant (k_c = 4)
            F_ab = np.array([F * np.cos(theta), F * np.sin(theta)])
            F_ba = -F_ab

            agent_a.add_force(F_ba)
            agent_b.add_force(F_ab)

        # integrate dynamics
        for agent in self.agent_list:
            agent.apply_net_force()


    def simulate_robotized(self):
        if len(self.agent_list) == 2:
            agent_a = self.agent_list[0] # the robot particle
            agent_b = self.agent_list[1] # the other particle
            num_iters = len(agent_a.x_history)

            for i in range(num_iters):
                r = np.linalg.norm([agent_a.x_history[i] - agent_b.x, agent_a.y_history[i] - agent_b.y])
                theta = np.arctan2(agent_b.y - agent_a.y_history[i], agent_b.x - agent_a.x_history[i])
                F =  4 * agent_a.charge * agent_b.charge / r**2 # based on Coulomb force with modified Coulomb constant (k_c = 4)
                F_ab = np.array([F * np.cos(theta), F * np.sin(theta)])

                agent_b.add_force(F_ab)

                # integrate dynamics
                agent_b.apply_net_force()
        elif len(self.agent_list) == 3:
            agent_a = self.agent_list[0] # the robot particle
            agent_b = self.agent_list[1] # second particle
            agent_c = self.agent_list[2] # third particle
            num_iters = len(agent_a.x_history) - 1

            for i in range(num_iters):
                r_ab = np.linalg.norm([agent_a.x_history[i] - agent_b.x, agent_a.y_history[i] - agent_b.y])
                r_ac = np.linalg.norm([agent_a.x_history[i] - agent_c.x, agent_a.y_history[i] - agent_c.y])
                r_bc = np.linalg.norm([agent_b.x - agent_c.x, agent_b.y - agent_c.y])
                theta_ab = np.arctan2(agent_b.y - agent_a.y_history[i], agent_b.x - agent_a.x_history[i])
                theta_ac = np.arctan2(agent_c.y - agent_a.y_history[i], agent_c.x - agent_a.x_history[i])
                theta_bc = np.arctan2(agent_c.y - agent_b.y, agent_c.x - agent_c.x)
                f_ab =  4 * agent_a.charge * agent_b.charge / r_ab**2 # based on Coulomb force with modified Coulomb constant (k_c = 4)
                f_ac =  4 * agent_a.charge * agent_c.charge / r_ac**2 # based on Coulomb force with modified Coulomb constant (k_c = 4)
                f_bc =  4 * agent_b.charge * agent_c.charge / r_bc**2 # based on Coulomb force with modified Coulomb constant (k_c = 4)
                F_ab = np.array([f_ab * np.cos(theta_ab), f_ab * np.sin(theta_ab)])
                F_ac = np.array([f_ac * np.cos(theta_ac), f_ac * np.sin(theta_ac)])
                F_bc = np.array([f_bc * np.cos(theta_bc), f_bc * np.sin(theta_bc)])

                agent_b.add_force(F_ab)
                agent_b.add_force(-F_bc)
                agent_c.add_force(F_ac)
                agent_c.add_force(F_bc)


                # integrate dynamics
                agent_b.apply_net_force()
                agent_c.apply_net_force()


    def simulate_run(self, t=3, dt=0.001, plot=False):
        iters = int(t // dt)

        for i in range(iters):
            self.simulate_iteration()
            if i % 50 == 0 and plot:
                self.plot(i)


    def export_data(self):
        scenario_data = []
        for agent in self.agent_list:
            scenario_data.append(agent.history())

        return scenario_data


    def plot(self, iter_num):
        plt.figure()
        ax = plt.gca()

        # reference circle
        circle = plt.Circle((0, 0), self.env_radius, linestyle='--', color='k', fill=False)
        ax.add_artist(circle)

        # plot agents
        for a in self.agent_list:
            circle = plt.Circle((a.x, a.y), a.radius, linestyle='-', color='b', fill=True)
            ax.add_artist(circle)

        # formatting
        ax_lim = 1.5*np.array([-self.env_radius, self.env_radius])
        plt.xlim(ax_lim)
        plt.ylim(ax_lim)
        ax.set_aspect(1.0)

        plt.savefig("figs/" + str(iter_num) + ".png", dpi=300)
        plt.close()


def generate_dataset(filename, num_runs=1000, augment_factor=4, num_agents=2, env_radius=10):
    dataset = []
    for _ in trange(num_runs):
        scenario = Scenario(num_agents, env_radius=env_radius)
        scenario.simulate_run()
        # dataset.append(scenario.export_data()) # commented to eliminate Newton's 3rd Law examples

        for j in range(augment_factor):
            scenario.reset_other_agent()
            scenario.simulate_robotized()
            dataset.append(scenario.export_data())

    with open(filename, 'wb') as f:
        dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL)

    return dataset


def main():
    generate_dataset("train_data_2_robot.pkl", num_runs=250)
    print("Train data generated.")
    generate_dataset("val_data_2_robot.pkl", num_runs=75)
    print("Validation data generated.")
    generate_dataset("test_data_2_robot.pkl", num_runs=50)
    print("Test data generated.")


if __name__ == "__main__":
    main()
