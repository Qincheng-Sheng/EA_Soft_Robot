# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:18:51 2022

@author: 11027
"""

import concurrent.futures

from vpython import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import random as rd
import time
import pickle


class mass:
    def __init__(self, m, p, v, a, ID):
        self.m = m  # mass
        self.p = p  # position
        self.v = v  # velocity
        self.a = a  # acceleration
        self.id = ID


class spring:
    def __init__(self, k, l, c, a, b, rst):
        self.k = k  # spring constant
        self.l = l  # current rest length
        self.c = c  # indices of two connected masses by this spring
        self.a = a  # amplify factor for the expression a * sin(wt + b)
        self.b = b  # offset factor for the expression a * sin(wt + b)
        self.rest = rst  # the initial rest length of spring


class infomation:
    def __init__(self, pts, connects, ids, center, l):
        self.point_list = pts  # record of position of all masses
        self.connect_list = connects  # record of connection relation of springs
        self.id_list = ids  # record of id of masses
        self.center_list = center  # list of center position of cubes
        self.length = l  # length of cube


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
# L0 = 0.1  # length of cube
damping = 0.9999
friction_mu_s = 1
friction_mu_k = 0.8
ts = 1500
evaluation_num = 200

global_k = [1000, 20000, 5000, 5000]
global_w = np.pi * 2 / 500
global_a = [0, 0, 0.25, 0.25]
global_b = [0, 0, 0, np.pi]

'''
global_k = [1000, 1000, 1000, 1000]
global_w = np.pi * 2 / 400
global_a = [0, 0, 0., 0.]
global_b = [0, 0, 0, np.pi]
'''

def grid_ground(xmax, num):
    # xmax = extent of grid in each direction
    # num = number of lines

    # create vertical lines:
    l = np.linspace(-xmax, xmax, num=num)
    for i in l:
        curve(pos=[vector(i, xmax, 0), vector(i, -xmax, 0)])
        curve(pos=[vector(xmax, i, 0), vector(-xmax, i, 0)])
    return


def spring_Info(masses, id1, id2):
    mass_idList = []
    for ms in masses:
        mass_idList.append(ms.id)
    i = mass_idList.index(id1)
    j = mass_idList.index(id2)
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

def get_cube(cube, masses, springs, info, material_index):
    # x,y,z is the center of the cube, l is the cube length
    info.center_list.append(cube)
    l = info.length

    x = cube[0]
    y = cube[1]
    z = cube[2]
    pos = [[x + l / 2, y + l / 2, z + l / 2], [x + l / 2, y - l / 2, z + l / 2], [x - l / 2, y + l / 2, z + l / 2],
           [x - l / 2, y - l / 2, z + l / 2],
           [x + l / 2, y + l / 2, z - l / 2], [x + l / 2, y - l / 2, z - l / 2], [x - l / 2, y + l / 2, z - l / 2],
           [x - l / 2, y - l / 2, z - l / 2], ]
    for p in pos:
        p[0] = round(p[0], 4)  # remove the floating error
        p[1] = round(p[1], 4)
        p[2] = round(p[2], 4)

    if len(info.id_list) > 0:
        max_id = np.max(info.id_list)
    else:
        max_id = -1
    this_cube = []  # mass list of this new cube (can include old mass point)
    if info.point_list:  # if it is not the first cube
        for i in range(len(pos)):  # for every pos[i], we have to know if it is already in the point_list
            if pos[i] in info.point_list:  # find the index of the exist mass point in the mass list
                this_cube.append(masses[info.point_list.index(pos[i])])  # get this mass in this_cube, but not masses,
                # because we already have it!
            else:  # if this pos[i] is not in the point_list at all, we just add it as new mass point
                new_id = max_id + 1 + i
                new_mass = mass(m, pos[i], [0, 0, 0], [0, 0, 0], new_id)

                masses.append(new_mass)  # add this mass in this_cube, also in masses
                this_cube.append(new_mass)

                info.id_list.append(new_id)
                info.point_list.append(pos[i])

    else:  # create first cube
        for i in range(len(pos)):
            new_id = max_id + 1 + i
            new_mass = mass(m, pos[i], [0, 0, 0], [0, 0, 0], new_id)

            masses.append(new_mass)
            this_cube.append(new_mass)

            info.id_list.append(new_id)
            info.point_list.append(pos[i])
    count = -1

    for i in range(8):
        for j in range(8):
            if i < j:
                count = count + 1
                connection = sorted([this_cube[i].id, this_cube[j].id])
                if connection not in info.connect_list:
                    l, initial_vec, mid_point = spring_Info(this_cube, connection[0], connection[1])
                    k, a, b = global_k[material_index], global_a[material_index], global_b[material_index]
                    springs.append(spring(k, l, connection, a, b, l))
                    info.connect_list.append(connection)

    return masses, springs, info


def vis_cube(masses, info):
    spheres = []  # a list of spheres
    sides = []  # a list of cylinder

    for i, pos_item in enumerate(masses):
        spheres.append(sphere(pos=vector(masses[i].p[0], masses[i].p[1],
                                         masses[i].p[2]), radius=r1, color=color.red))

    for connection in info.connect_list:
        id1 = connection[0]
        id2 = connection[1]
        mass1 = masses[info.id_list.index(id1)]
        l, initial_vec, mid_point = spring_Info(masses, id1, id2)

        sides.append(
            cylinder(pos=vector(mass1.p[0], mass1.p[1], mass1.p[2], ),
                     axis=vector(initial_vec[0], initial_vec[1],
                                 initial_vec[2]),
                     color=color.blue, radius=r2))
    return spheres, sides


def update_spring(spheres, springs, sides, id_list):
    for i, spr in enumerate(springs):
        m1, m2 = spr.c[0], spr.c[1]
        m1 = id_list.index(m1)
        m2 = id_list.index(m2)
        sphere1 = spheres[m1]
        sphere2 = spheres[m2]
        sides[i].pos = sphere1.pos
        sides[i].axis = sphere2.pos - sphere1.pos
    return sides


def getcenter(masses):
    center = [0, 0, 0]
    mass_num = len(masses)
    for ms in masses:
        center[0] += ms.p[0]
        center[1] += ms.p[1]
        center[2] += ms.p[2]

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
    #fitness = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    # bouncing fitness
    fitness = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)/np.abs(10*z0)**2
    #fitness = (x0 - x1)

    return fitness


def vis_run_model(ts, robot):
    #robot =  [masses_v, springs_v, spheres_v, sides_v,info_v]
    masses = robot[0]
    springs = robot[1]
    mass_idList = robot[2].id_list
    spheres = robot[3]
    sides = robot[4]

    T = 0
    mass_count = len(masses)
    g_center = getcenter(masses)
    fitness = 0
    for i in range(ts):
        rate(10000)
        # check each mass for its force
        for j in range(mass_count):
            # set up gravity
            force = masses[j].m * deepcopy(g)

            # check force caused by spring that connects to this mass point
            for _, edge in enumerate(springs):
                # -----
                # set parameters in expression L0 = L0_initial * [1 + a * sin(wt+ b)] for each spring.
                edge.l = edge.rest * (1 + edge.a * np.sin(global_w * i + edge.b))
                # -----
                if masses[j].id in edge.c:  # we always define the mass point we are checking now as start point
                    if edge.c[0] == masses[j].id:
                        start = edge.c[0]
                        end = edge.c[1]
                    else:
                        start = edge.c[1]
                        end = edge.c[0]

                    start = mass_idList.index(start)
                    end = mass_idList.index(end)
                    sphere1 = spheres[start]
                    sphere2 = spheres[end]
                    vect = sphere2.pos - sphere1.pos
                    unit_vec = norm(vect)
                    length = mag(vect)

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
                    force = force + F_N * friction_mu_k * (-unit_FP)

                # give restoration force (bounce)
                force += np.array([0, 0, -kc * masses[j].p[2]])

            masses[j].a = force / m
            dv = [masses[j].a[0] * dt, masses[j].a[1] * dt, masses[j].a[2] * dt]
            masses[j].v = [masses[j].v[0] + dv[0], masses[j].v[1] + dv[1], masses[j].v[2] + dv[2]]
            for v_i, v_value in enumerate(masses[j].v):
                masses[j].v[v_i] = v_value * damping
            dp = [masses[j].v[0] * dt, masses[j].v[1] * dt, masses[j].v[2] * dt]
            masses[j].p = [masses[j].p[0] + dp[0], masses[j].p[1] + dp[1], masses[j].p[2] + dp[2]]
            # print(masses[j].p[2])
        for t in range(mass_count):
            # update the position of sphere according to the position change of mass point
            spheres[t].pos = vector(masses[t].p[0], masses[t].p[1], masses[t].p[2])
        #print("height: ", spheres[1].pos.z)

        # update the position of cylinder according to the position change of mass and length change of spring
        sides = update_spring(spheres, springs, sides, mass_idList)

        g_center_new = getcenter(masses)
        fitness += get_fitness(g_center_new, g_center)
        g_center = g_center_new
        T += dt
        print('time_step', i)
        print("fitness: ",fitness)

    return fitness

def run_model(ts, robot):
    #robot =  [masses_v, springs_v, spheres_v, sides_v,info_v]
    masses = robot[0]
    springs = robot[1]
    #spheres = robot[2]
    #sides = robot[3]
    mass_idList = robot[2].id_list
    T = 0
    mass_count = len(masses)
    g_center = getcenter(masses)
    fitness = 0
    for i in range(ts):
        rate(10000)
        # check each mass for its force
        for j in range(mass_count):
            # set up gravity
            force = masses[j].m * deepcopy(g)

            # check force caused by spring that connects to this mass point
            for _, edge in enumerate(springs):
                # -----
                # set parameters in expression L0 = L0_initial * [1 + a * sin(wt+ b)] for each spring.
                edge.l = edge.rest * (1 + edge.a * np.sin(global_w * i + edge.b))
                # -----
                if masses[j].id in edge.c:  # we always define the mass point we are checking now as start point
                    if edge.c[0] == masses[j].id:
                        start = edge.c[0]
                        end = edge.c[1]
                    else:
                        start = edge.c[1]
                        end = edge.c[0]

                    start = mass_idList.index(start)
                    end = mass_idList.index(end)
                    sphere1 = masses[start]
                    sphere2 = masses[end]
                    vect = np.array(sphere2.p) - np.array(sphere1.p)
                    length = np.linalg.norm(vect)
                    unit_vec = vect / length

                    F = edge.k * (length - edge.l)
                    # add up the force from each spring that connects to this mass point j.
                    force = force + F * unit_vec
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
                    force = force + F_N * friction_mu_k * (-unit_FP)

                # give restoration force (bounce)
                force += np.array([0, 0, -kc * masses[j].p[2]])

            masses[j].a = force / m
            dv = [masses[j].a[0] * dt, masses[j].a[1] * dt, masses[j].a[2] * dt]
            masses[j].v = [masses[j].v[0] + dv[0], masses[j].v[1] + dv[1], masses[j].v[2] + dv[2]]
            for v_i, v_value in enumerate(masses[j].v):
                masses[j].v[v_i] = v_value * damping
            dp = [masses[j].v[0] * dt, masses[j].v[1] * dt, masses[j].v[2] * dt]
            masses[j].p = [masses[j].p[0] + dp[0], masses[j].p[1] + dp[1], masses[j].p[2] + dp[2]]
            # print(masses[j].p[2])
        #for t in range(mass_count):
            # update the position of sphere according to the position change of mass point
        #    spheres[t].pos = vector(masses[t].p[0], masses[t].p[1], masses[t].p[2])
        #print("height: ", spheres[1].pos.z)

        # update the position of cylinder according to the position change of mass and length change of spring
        #sides = update_spring(spheres, springs, sides, mass_idList)

        g_center_new = getcenter(masses)
        fitness += get_fitness(g_center_new, g_center)
        g_center = g_center_new
        T += dt
        if i % 100 == 0:
            print('time step:', i)
            print("fitness: ",fitness)

    return fitness

def build_robot(centers_and_material,  visualize=False):
    masses = []
    springs = []
    info = infomation([], [], [], [], 0.1)

    for c_m in centers_and_material:
        #print(c_m)
        masses, springs, info = get_cube(c_m[0], masses, springs, info,c_m[1])

    #print("number of mass:", len(masses))
    #print("number of spring:", len(springs))

    if visualize:
        canvas(width=1200, height=600, center=vector(0, 0, 0), background=color.white)
        grid_ground(1, 20)
        spheres, sides = vis_cube(masses, info)

        # base[0]:masses, base[1]:springs, base[2]:info,  vis_cube()=> spheres, sides

        return masses, springs, info, spheres, sides
    return masses, springs, info, [], []

def nodeCount(x):
    if "children" not in x:
        return 1
    return sum([nodeCount(c) for c in x["children"]]) + 1

def get_all_center(node):
    centers = [[node['center'],node['material']]]
    if node["children"] == []:
        return [[node['center'],node['material']]]
    else:
        for child in node["children"]:
            centers = centers + get_all_center(child)
        return centers

def select_random_node(selected, depth):
    if not selected["children"]: #if there is no children in this node
        return selected
    # favor nodes near the root
    if rd.randint(0, 6) < 2 * depth:
        while True:
            choice = rd.randint(0, len(faces) - 1)
            grow = faces[choice]
            if selected[grow] == None:
                return selected

    child_count = len(selected["children"])
    return select_random_node(selected["children"][rd.randint(0, child_count - 1)], depth + 1)

# in the robot tree, find a random node and add a new cube under it
def add_cube(robot):
    # we don't want too many cube
    pick_cube = select_random_node(robot, 0)
    while True:
        choice = rd.randint(0, len(faces) - 1)
        grow = faces[choice]

        if pick_cube[grow] == None:
            direction = grow_direction[grow]
            new_center = [round(pick_cube['center'][0] + direction[0], 4),
                          round(pick_cube['center'][1] + direction[1], 4),
                          round(pick_cube['center'][2] + direction[2], 4), ]
            if round(new_center[2], 4) <= 0:
                #print("skip: underground cube!")
                return robot
            new_cube = {'center': new_center,'material':rd.randint(0,3), 'children': [], "up": None, "down": None, "front": None, "back": None,
                        "left": None,
                        "right": None}
            pick_cube['children'].append(new_cube)
            pick_cube[grow] = new_cube
            return robot

cube_center0 = [0, 0, 0.05]  #  indicate the position of the base cube


cube0 = {'center': cube_center0,'material':rd.randint(0,3), 'children': [], "up": None, "down": None, "front": None, "back": None, "left": None,
         "right": None}

grow_step = 0.1
faces = ["up", "down", "front", "back", "left", "right"]
grow_direction = {"up": [0, 0, grow_step], "down": [0, 0, -grow_step], "front": [grow_step, 0, 0],
                  "back": [-grow_step, 0, 0], "left": [0, grow_step, 0], "right": [0, -grow_step, 0]}


def swap(i1, i2):
    temp = i1
    i1 = i2
    i2 = temp

    return i1, i2

def xover(r1,r2):
    child_r1 = deepcopy(r1)
    child_r2 = deepcopy(r2)
    choice = rd.randint(0, len(faces) - 1)
    exchange_face = faces[choice]
    #print(faces[choice])

    branch1 = child_r1[exchange_face]
    branch2 = child_r2[exchange_face]
    if branch1 != None and branch2 != None:
        child_r1['children'].remove(branch1)
        child_r2['children'].remove(branch2)
        child_r1['children'].append(branch2)
        child_r2['children'].append(branch1)

    elif  branch1 == None and branch2 != None:
        child_r2['children'].remove(branch2)
        child_r1['children'].append(branch2)

    elif  branch1 != None and branch2 == None:
        child_r1['children'].remove(branch1)
        child_r2['children'].append(branch1)

    branch1, branch2 = swap(branch1, branch2)
    child_r1[exchange_face] = branch1
    child_r2[exchange_face] = branch2
    # if child robot is too large, we don't want it. replace that child with original parent

    return child_r1, child_r2

def mutation(r0):
    # 50% change the material of cube, else add one more cube
    if rd.random()>0.5:
        pick_cube = select_random_node(r0, 0)
        pick_cube['material'] = rd.randint(0,3)

    else:
        r0 = add_cube(r0)
    return r0

def evolution(robots_trees,population_fitness,pick_indices):
    random_parent1, random_parent2 = pick_indices
    c1 = deepcopy(robots_trees[random_parent1])
    c2 = deepcopy(robots_trees[random_parent2])
    c1, c2 = xover(c1, c2)

    # do mutation
    c1 = mutation(c1)
    c2 = mutation(c2)

    # evaluate offspring using parallel coding
    children_robots_trees = [c1, c2]
    children_cube_centers_and_materials = []

    ts_parallel = [ts for _ in range(2)]
    offspring_fitness = []
    children_robot_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for c_m in executor.map(get_all_center, children_robots_trees):
            children_cube_centers_and_materials.append(c_m)

        for robot in executor.map(build_robot, children_cube_centers_and_materials):
            robot_mass = robot[0]
            robot_spring = robot[1]
            robot_info = robot[2]
            children_robot_list.append([robot_mass, robot_spring, robot_info])

        for f in executor.map(run_model, ts_parallel, children_robot_list):
            offspring_fitness.append(f)

    c1_fit = offspring_fitness[0]
    c2_fit = offspring_fitness[1]

    # compare and select
    p1_fit = population_fitness[random_parent1]
    p2_fit = population_fitness[random_parent2]
    if c1_fit > c2_fit:
        # replace two parents when c1>c2 and c2> any of p
        if c2_fit > p1_fit and c2_fit > p2_fit:
            robots_trees[random_parent1] = c1
            robots_trees[random_parent2] = c2
            population_fitness[random_parent1] = c1_fit
            population_fitness[random_parent2] = c2_fit

        elif c2_fit > p1_fit:
            if c1_fit > p2_fit:
                # replace two parents when c2>p1 and c1>p2
                robots_trees[random_parent1] = c1
                robots_trees[random_parent2] = c2
                population_fitness[random_parent1] = c1_fit
                population_fitness[random_parent2] = c2_fit
            else:
                # replace one parent when c2>p1, c1<p2 and c1>c2
                robots_trees[random_parent1] = c1
                population_fitness[random_parent1] = c1_fit
        elif c2_fit > p2_fit:
            if c1_fit > p1_fit:
                # replace two parents when c2>p2 and c1>p1
                robots_trees[random_parent1] = c1
                robots_trees[random_parent2] = c2
                population_fitness[random_parent1] = c1_fit
                population_fitness[random_parent2] = c2_fit
            else:
                # replace one parent when c2>p2, c1<p1 and c1>c2
                robots_trees[random_parent2] = c1
                population_fitness[random_parent2] = c1_fit
        else:
            # c2 is smaller than both p1 and p2, so we only consider c1
            if p1_fit > p2_fit:
                if c1_fit > p2_fit:
                    # replace p2 when c1>p2 and p1>p2
                    robots_trees[random_parent2] = c1
                    population_fitness[random_parent2] = c1_fit

                # we do nothing is c1 is smaller than both p1 and p2

            else:
                if c1_fit > p1_fit:
                    # replace p1 when c1>p1 and p2>p1
                    robots_trees[random_parent1] = c1
                    population_fitness[random_parent1] = c1_fit

    else:
        # replace two parents when c2>c1 and c1> any of p
        if c1_fit > p1_fit and c1_fit > p2_fit:
            robots_trees[random_parent1] = c1
            robots_trees[random_parent2] = c2
            population_fitness[random_parent1] = c1_fit
            population_fitness[random_parent2] = c2_fit

        elif c1_fit > p1_fit:
            if c2_fit > p2_fit:
                # replace two parents when c1>p1 and c2>p2
                robots_trees[random_parent1] = c1
                robots_trees[random_parent2] = c2
                population_fitness[random_parent1] = c1_fit
                population_fitness[random_parent2] = c2_fit
            else:
                # replace one parent when c1>p1, c2<p2 and c2>c1
                robots_trees[random_parent1] = c2
                population_fitness[random_parent1] = c2_fit
        elif c1_fit > p2_fit:
            if c2_fit > p1_fit:
                # replace two parents when c1>p2 and c2>p1
                robots_trees[random_parent1] = c1
                robots_trees[random_parent2] = c2
                population_fitness[random_parent1] = c1_fit
                population_fitness[random_parent2] = c2_fit
            else:
                # replace one parent when c1>p2, c2<p1 and c2>c1
                robots_trees[random_parent2] = c2
                population_fitness[random_parent2] = c2_fit
        else:
            # c1 is smaller than both p1 and p2, so we only consider c1
            if p1_fit > p2_fit:
                if c2_fit > p2_fit:
                    # replace p2 when c2>p2 and p1>p2
                    robots_trees[random_parent2] = c2
                    population_fitness[random_parent2] = c2_fit

                # we do nothing is c1 is smaller than both p1 and p2

            else:
                if c2_fit > p1_fit:
                    # replace p1 when c2>p1 and p2>p1
                    robots_trees[random_parent1] = c2
                    population_fitness[random_parent1] = c2_fit
    return robots_trees[random_parent1],robots_trees[random_parent2], population_fitness[random_parent1],population_fitness[random_parent1],pick_indices


def more_cube_robot(r, num):
    for _ in range(num):
        r = add_cube(r)
    return r

if __name__ == '__main__':
    # random search
    print("start random search:")

    random_fitness = []
    robots_list_rd = []
    robots_trees_random = []
    robots_trees_random_after=[]
    for _ in range(evaluation_num):
        robots_trees_random.append(deepcopy(cube0))
    cube_num_list = [rd.randint(0,5) for _ in range(evaluation_num)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for rd_trees in executor.map(more_cube_robot, robots_trees_random,cube_num_list):
            robots_trees_random_after.append(rd_trees)

    cube_centers_and_materials_rd = []
    ts_parallel = [ts for _ in range(evaluation_num)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for c_m in executor.map(get_all_center, robots_trees_random_after):
            cube_centers_and_materials_rd.append(c_m)

        for robot in executor.map(build_robot, cube_centers_and_materials_rd):
            robot_mass = robot[0]
            robot_spring = robot[1]
            robot_info = robot[2]
            robots_list_rd.append([robot_mass, robot_spring, robot_info])

        for f in executor.map(run_model, ts_parallel, robots_list_rd):
            random_fitness.append(f)



    # hill climber
    print("start hill climber:")
    robot_hc = deepcopy(cube0)
    hill_fitness = []
    best_robot_tree_hc = robot_hc

    for _ in range(evaluation_num):
        start_time = time.time()
        cube_centers_and_materials_hc = get_all_center(robot_hc)

        masses_hc, springs_hc, info_hc, _, _ = build_robot(cube_centers_and_materials_hc, visualize=False)
        h_fit = run_model(ts, [masses_hc, springs_hc, info_hc, ])
        if len(hill_fitness) > 0:
            if h_fit > hill_fitness[-1]:
                hill_fitness.append(h_fit)
                best_robot_tree_hc = robot_hc
            else:
                hill_fitness.append(hill_fitness[-1])
                robot_hc = best_robot_tree_hc
        else:
            hill_fitness.append(h_fit)
        robot_hc = mutation(robot_hc)
        print("it cost {} seconds in this evaluation".format(time.time() - start_time))
        



    print("start genetic algorithm:")
    population = 20

    robots_trees = []
    for _ in range(population):
        robots_trees.append(deepcopy(cube0))

    GA_fitness = []
    best_robot_GA = None


    robot_info_list = []
    population_fitness = []

    # parallel coding
    robot_list = []
    cube_centers_and_materials = []
    ts_parallel = [ts for _ in range(population)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for c_m in executor.map(get_all_center, robots_trees):
            cube_centers_and_materials.append(c_m)

        for robot in executor.map(build_robot, cube_centers_and_materials):
            robot_mass = robot[0]
            robot_spring = robot[1]
            robot_info = robot[2]
            robot_list.append([robot_mass, robot_spring, robot_info])

        for f in executor.map(run_model, ts_parallel, robot_list):
            population_fitness.append(f)

    for e in range(evaluation_num):
        start_time = time.time()

        pick_list = rd.sample(range(population), int(population/2))
        couple = []
        while pick_list!= []:
            couple1 = pick_list[0]
            del (pick_list[0])
            couple2 = pick_list[0]
            del (pick_list[0])
            couple.append([couple1, couple2])
        robots_trees_parallel = [robots_trees for _ in range(int(population/2))]
        population_fitness_parallel = [population_fitness for _ in range(int(population / 2))]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for r_trees0, r_trees1,p_fitness0,p_fitness1,couple_indices in executor.map(evolution, robots_trees_parallel,population_fitness_parallel,couple):
                robots_trees[couple_indices[0]] = r_trees0
                robots_trees[couple_indices[1]] = r_trees1
                population_fitness[couple_indices[0]] = p_fitness0
                population_fitness[couple_indices[1]] = p_fitness1

        # find best fitness and best robot trees after selection
        best_idx = np.argmax(population_fitness)
        GA_fitness.append(population_fitness[best_idx])
        best_tree_GA = robots_trees[best_idx]
        with open('robot_trees_data_{}.pkl'.format(e), 'wb') as outp:
            for tree in robots_trees:
                pickle.dump(tree, outp, pickle.HIGHEST_PROTOCOL)

        print("it cost {} seconds in the evaluation {}".format(time.time() - start_time, e))

    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(range(evaluation_num), random_fitness, label="random search")
    axes.plot(range(evaluation_num), hill_fitness, label="hill climber")
    axes.plot(range(evaluation_num), GA_fitness, label="genetic algorithm")
    axes.set_xlabel('evaluation')
    axes.set_ylabel('fitness')
    axes.set_title('Learning Curve')
    plt.legend(loc='best')
    plt.show()

    cube_centers_and_materials1 = get_all_center(best_tree_GA)
    masses1, springs1, info1, _, _ = build_robot(cube_centers_and_materials1, visualize='True')
    canvas(width=1200, height=600, center=vector(0, 0, 0), background=color.white)
    grid_ground(1, 20)
    start_time = time.time()
    spheres1, sides1 = vis_cube(masses1, info1)
    vis_run_model(ts, [masses1, springs1, info1, spheres1, sides1])
