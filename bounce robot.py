from vpython import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import random as rd
import time


r_region = [[-0.05593587778529305, -0.18538955968658374, 0.06811762788875703, 0], [-0.18694923177480818, -0.10956516762575688, 0.08883104759401583, 1], [-0.07105263189984004, 0.1549828936677683, 0.06369918024767823, 2], [-0.1803907462182595, 0.18573522625563638, 0.04785276971903682, 3]]
best_region_HC = [[0.049074607660021335, 0.17141236217420835, -0.15569096810395108, 0], [0.09761007718061926, 0.014742665389432252, 0.14387855536384309, 1], [0.04851473788703897, 0.10388045359467485, -0.11790789970144973, 2], [0.14365837350192323, -0.19743331823007157, 0.1673312739886827, 3]]
best_region_GA = best_region_HC

class mass:
    def __init__(self, m, p, v, a):
        self.m = m  # mass
        self.p = p  # position
        self.v = v  # velocity
        self.a = a  # acceleration


class spring:
    def __init__(self, k, l, c, a, b):
        self.k = k  # spring constant
        self.l = l  # rest length
        self.c = c  # indices of two connected masses by this spring
        self.a = a  # amplify factor for the expression a * sin(wt + b)
        self.b = b  # offset factor for the expression a * sin(wt + b)


# initial condition
r1 = 0.01  # radius of the ball
r2 = 0.005  # radius of the cylinder
h = 0.0  # the initial height
g_const = 9.81
g = np.array([0, 0, - g_const])  # gravity vector
# g = np.array([0, 0, 0])  # test zero gravity
dt = 0.0001  # time step
m = 0.1  # corner mass
T = 0  # global time variable
# k = 5000  # spring constant of the spring
kc = 100000  # spring constant of the ground
L0 = 0.1  # length of cube
damping = 0.9999
friction_mu_s = 1
friction_mu_k = 0.8
ts = 600


def grid_ground(xmax, num):
    # xmax = extent of grid in each direction
    # num = number of lines

    # create vertical lines:
    l = np.linspace(-xmax, xmax, num=num)
    for i in l:
        curve(pos=[vector(i, xmax, 0), vector(i, -xmax, 0)])
        curve(pos=[vector(xmax, i, 0), vector(-xmax, i, 0)])
    return


## create the world
# canvas(width=1200, height=600, center=vector(0, 0, 0), background=color.white)
# grid_ground(1, 20)


def set_height(h):
    initial_position = [[0, 0, h], [0.1, 0, h], [0, 0.1, h], [0.1, 0.1, h],
                        [0, 0, h + 0.1], [0.1, 0, h + 0.1], [0, 0.1, h + 0.1],
                        [0.1, 0.1, h + 0.1], [0.2, 0, h], [0.2, 0.1, h],
                        [0.2, 0, h + 0.1], [0.2, 0.1, h + 0.1], [0.1, 0.2, h],
                        [0, 0.2, h], [0.1, 0.2, h + 0.1], [0, 0.2, h + 0.1], [-0.1, 0, h],
                        [-0.1, 0.1, h], [-0.1, 0, h + 0.1], [-0.1, 0.1, h + 0.1], [0.1, -0.1, h],
                        [0, -0.1, h], [0.1, -0.1, h + 0.1], [0, -0.1, h + 0.1]]

    return initial_position


# compute the rest of length of a spring and its vector between masses i and j
def spring_Info(masses, i, j):
    x_i = masses[i].p[0]
    y_i = masses[i].p[1]
    z_i = masses[i].p[2]
    x_j = masses[j].p[0]
    y_j = masses[j].p[1]
    z_j = masses[j].p[2]

    vectors = [x_j - x_i,
               y_j - y_i,
               z_j - z_i]

    rest_of_length = np.linalg.norm(vectors)

    mid_point = [(x_j + x_i) / 2,
                 (y_j + y_i) / 2,
                 (z_j + z_i) / 2]

    return rest_of_length, vectors, mid_point


def set_material(region, spring_midpoint):
    dist0 = np.linalg.norm(np.array([region[0][0], region[0][1], region[0][2]]) - np.array(spring_midpoint))
    dist1 = np.linalg.norm(np.array([region[1][0], region[1][1], region[1][2]]) - np.array(spring_midpoint))
    dist2 = np.linalg.norm(np.array([region[2][0], region[2][1], region[2][2]]) - np.array(spring_midpoint))
    dist3 = np.linalg.norm(np.array([region[3][0], region[3][1], region[3][2]]) - np.array(spring_midpoint))

    dist_list = [dist0, dist1, dist2, dist3]
    material_index = region[np.argmin(dist_list)][3]
    # print("this spring is of material:", material_index)

    return global_k[material_index], global_a[material_index], global_b[material_index]

def update_spring(spheres, springs, sides):
    for i, spr in enumerate(springs):
        m1, m2 = spr.c[0], spr.c[1]
        sphere1 = spheres[m1]
        sphere2 = spheres[m2]
        sides[i].pos = sphere1.pos
        sides[i].axis = sphere2.pos - sphere1.pos
    return sides


def getcenter(masses):
    center = [0, 0, 0]
    mass_num = len(masses)
    for m in masses:
        center[0] += m.p[0]
        center[1] += m.p[1]
        center[2] += m.p[2]

    center[0] = round(center[0] / mass_num, 4)
    center[1] = round(center[1] / mass_num, 4)
    center[2] = round(center[2] / mass_num, 4)

    return center


def get_fitness(gravity_center0, gravity_center1):
    x0 = gravity_center0[0]
    y0 = gravity_center0[1]
    z0 = gravity_center0[2]
    x1 = gravity_center1[0]
    y1 = gravity_center1[1]
    z1 = gravity_center1[2]

    # moving fitness
    fitness = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    # bouncing fitness
    # fitness = np.abs(z1-z0)
    
    if np.abs(z1) > 0.15:
        fitness = fitness-0.2*np.abs(z1)
    return fitness


global_k = [1000, 20000, 5000, 5000]
global_w = np.pi*2/200
global_a = [0, 0, 0.10, 0.10]
global_b = [0, 0, 0, np.pi]

def vis_init_robot(m, r1, r2, h, region):
    masses = []  # a list of mass
    spheres = []  # a list of spheres

    initial_pos = set_height(h)

    for i, pos_item in enumerate(initial_pos):
        masses.append(mass(m, pos_item, [0, 0, 0], [0, 0, 0]))  # cube corners
        spheres.append(sphere(pos=vector(masses[i].p[0], masses[i].p[1],
                                         masses[i].p[2]), radius=r1, color=color.red))

    springs = []  # a list of spring object
    sides = []  # a list of cylinder
    spring_lengthList = []

    count = -1
    for i in range(8):
        for j in range(8):
            if i < j:
                count = count + 1
                l, initial_vec, mid_point = spring_Info(masses, i, j)
                spring_lengthList.append(l)
                k, a, b = set_material(region, mid_point)
                springs.append(spring(k, l, [i, j], a, b))
                sides.append(
                    cylinder(pos=spheres[i].pos,
                             axis=vector(initial_vec[0], initial_vec[1],
                                         initial_vec[2]),
                             color=color.blue, radius=r2))

                # build other four cube
    vec_list1 = [1, 3, 5, 7, 8, 9, 10, 11]
    for i, item in enumerate(vec_list1):
        for j in range(8, 12):
            if item < j:
                count = count + 1
                l, initial_vec, mid_point = spring_Info(masses, item, j)
                spring_lengthList.append(l)
                k, a, b = set_material(region, mid_point)
                springs.append(spring(k, l, [item, j], a, b))
                sides.append(
                    cylinder(pos=spheres[item].pos,
                             axis=vector(initial_vec[0], initial_vec[1],
                                         initial_vec[2]),
                             color=color.blue, radius=r2))

    vec_list2 = [0, 2, 4, 6, 16, 17, 18, 19]
    for i, item in enumerate(vec_list2):
        for j in range(16, 20):
            if item < j:
                count = count + 1
                l, initial_vec, mid_point = spring_Info(masses, item, j)
                spring_lengthList.append(l)
                k, a, b = set_material(region, mid_point)
                springs.append(spring(k, l, [item, j], a, b))
                sides.append(
                    cylinder(pos=spheres[item].pos,
                             axis=vector(initial_vec[0], initial_vec[1],
                                         initial_vec[2]),
                             color=color.blue, radius=r2))
    vec_list2 = [2, 3, 6, 7, 12, 13, 14, 15]
    for i, item in enumerate(vec_list2):
        for j in range(12, 16):
            if item < j:
                count = count + 1
                l, initial_vec, mid_point = spring_Info(masses, item, j)
                spring_lengthList.append(l)
                k, a, b = set_material(region, mid_point)
                springs.append(spring(k, l, [item, j], a, b))
                sides.append(
                    cylinder(pos=spheres[item].pos,
                             axis=vector(initial_vec[0], initial_vec[1],
                                         initial_vec[2]),
                             color=color.blue, radius=r2))

    vec_list3 = [0, 1, 4, 5, 20, 21, 22, 23]
    for i, item in enumerate(vec_list3):
        for j in range(20, 24):
            if item < j:
                count = count + 1
                l, initial_vec, mid_point = spring_Info(masses, item, j)
                spring_lengthList.append(l)
                k, a, b = set_material(region, mid_point)
                springs.append(spring(k, l, [item, j], a, b))
                sides.append(
                    cylinder(pos=spheres[item].pos,
                             axis=vector(initial_vec[0], initial_vec[1],
                                         initial_vec[2]),
                             color=color.blue, radius=r2))
    return masses, spheres, springs, sides, spring_lengthList

def vis_run_model(ts, masses, spheres, springs, sides,spring_lengthList, ):
    T = 0
    mass_count = len(masses)
    g_center = getcenter(masses)
    fitness = 0
    for i in range(ts):
        rate(10000)

        # check each mass for its force
        for j in range(mass_count):
            # set up gravity
            force = 19*masses[j].m * np.array([0,0,-9.81])

            # check force caused by spring that connects to this mass point
            for idx, edge in enumerate(springs):
                # -----
                # set parameters in expression L0 = L0_initial * [1 + a * sin(wt+ b)] for each spring.
                edge.l = spring_lengthList[idx] * ( 1 + edge.a * np.sin(global_w * i + edge.b))
                # -----
                if j in edge.c:  # we always define the mass point we are checking now as start point
                    if edge.c[0] == j:
                        start = edge.c[0]
                        end = edge.c[1]
                    else:
                        start = edge.c[1]
                        end = edge.c[0]

                    sphere1 = spheres[start]
                    sphere2 = spheres[end]
                    vec = sphere2.pos - sphere1.pos
                    unit_vec = norm(vec)
                    length = mag(vec)
                    F = edge.k * (length - edge.l)
                    # add up the force from each spring that connects to this mass point j.
                    force = force + F * np.array([unit_vec.x, unit_vec.y, unit_vec.z])
                    
            # check if the corner point hits the ground
            if masses[j].p[2] <= 0:
                # check friction
                F_P = np.array([force[0], force[1], 0])
                F_N = np.array([0, 0, force[2]])
                F_P_value = np.linalg.norm(F_P)
                F_N_value = np.linalg.norm(F_N)

                if F_P_value < F_N_value * friction_mu_s:
                    force = force - F_P
                else:
                    unit_FP = F_P / F_P_value
                    force = force - F_N_value * friction_mu_k * unit_FP

                # give restoration force (bounce)
                force += np.array([0, 0, -kc * masses[j].p[2]])

            masses[j].a = force / m

            dv = [masses[j].a[0] * dt, masses[j].a[1] * dt, masses[j].a[2] * dt]
            masses[j].v = [masses[j].v[0] + dv[0], masses[j].v[1] + dv[1], masses[j].v[2] + dv[2]]
            for v_i, v_value in enumerate(masses[j].v):
                masses[j].v[v_i] = v_value * damping

            dp = [masses[j].v[0] * dt, masses[j].v[1] * dt, masses[j].v[2] * dt]
            masses[j].p = [masses[j].p[0] + dp[0], masses[j].p[1] + dp[1], masses[j].p[2] + dp[2]]

        for t in range(mass_count):
            # update the position of sphere according to the position change of mass point
            spheres[t].pos = vector(masses[t].p[0], masses[t].p[1], masses[t].p[2])

        # update the position of cylinder according to the position change of mass and length change of spring
        sides = update_spring(spheres, springs, sides)
        g_center_new = getcenter(masses)
        fitness += get_fitness(g_center_new, g_center)
        g_center = g_center_new
        T += dt
        print('time_step', i)
        print(fitness)

    return fitness

canvas(width=1200, height=600, center=vector(0, 0, 0), background=color.white)
grid_ground(1, 40)
masses, spheres, springs, sides, spring_lengthList = vis_init_robot(m, r1, r2, 0.3, best_region_GA)
fitness = vis_run_model(5000, masses, spheres, springs, sides, spring_lengthList, )
