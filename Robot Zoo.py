from vpython import *
import random
import pickle
import numpy as np
from copy import deepcopy
from multiprocessing.dummy import Pool as ThreadPool
from joblib import Parallel, delayed

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
ts = 2000
global_k = [1000, 20000, 5000, 5000]
global_w = np.pi * 2 / 1000
global_a = [0, 0, 0.15, 0.15]
global_b = [0, 0, 0, np.pi]

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
    def __init__(self, pts,connects,ids,center,l):
        self.point_list = pts  # record of position of all masses
        self.connect_list = connects  # record of connection relation of springs
        self.id_list = ids  # record of id of masses
        self.center_list = center # list of center position of cubes
        self.length = l  # length of cube

def grid_ground(xmax, num):
    # xmax = extent of grid in each direction
    # num = number of lines

    # create vertical lines:
    l = np.linspace(-xmax, xmax, num=num)
    for i in l:
        curve(pos=[vector(i, xmax, 0), vector(i, -xmax, 0)])
        curve(pos=[vector(xmax, i, 0), vector(-xmax, i, 0)])


# compute the base six faces id
def compute_baseface(i):
    face_id = []
    if i > 0:
        face_list = ['top','bot','front','back','left','right']
        face_id = face_list[i-1]
    return face_id

# when we try to add a cube, compute what pos we could add a cube, accroding to the depth
# return the position where insert a cube
def compute_facerange(d):
    s = 1
    for i in range(d):
        s += 6*pow(5,d-i-1)
    i = random.randrange(s-6*pow(5,d-1),s) 
    return i

# connect the original face to other four mass points, comupte their mass id
# input: insert cube position, individual tree 
# output: changed face new mass id
def change_face(i,individual):
    face_id = []
    face2 = []
    face_id = individual[i]['face']
    face_list = ['top','bot','front','back','left','right']
    for i in range(6):
        if face_id == face_list[i] and i%2 == 0:
            face_id = face_list[i]+' '+face_list[i+1]
            face2.append(face_list[i+1])
        elif face_id == face_list[i] and i%2 == 1:
            face_id = face_list[i]+' '+face_list[i-1]
            face2.append(face_list[i-1])
    return face_id, face2

# compute the added five faces mass id
# input: insert cube position; new five faces range; original tree length;
#        the actual position to insert cube in tree L-k; individual tree
# output: new five faces mass id
def compute_face(face2,individual):
    face_list = ['top','bot','front','back','left','right']
    face_id = []
    for i in range(6):
        if face2 != face_list[i]: 
            face_id.append(face_list[i]) 
            
    return face_id

def set_material():
    p = random.randrange(1,5) 
    if p == 1: 
        spring_type = 'muscle1'
    elif p == 2:
        spring_type = 'bone'
    elif p == 3:
        spring_type = 'muscle2'
    else:
        spring_type = 'muscle3'
        
    return spring_type

def get_material(spring_type):
    global_k = [1000, 20000, 5000, 5000]
    global_w = np.pi * 2 / 1000
    global_a = [0, 0, 0.25, 0.25]
    global_b = [0, 0, 0, np.pi]
    
    if spring_type == 'bone':
        k= global_k[0]; a = global_a[0];b = global_b[0]
    elif spring_type == 'muscle1':
        k= global_k[1]; a = global_a[1];b = global_b[1]
    elif spring_type == 'muscle2':
        k= global_k[2]; a = global_a[2];b = global_b[2]
    elif spring_type == 'muscle3':
        k= global_k[3]; a = global_a[3];b = global_b[3]
    return k, a, b

# generate the population of the basic 8 masses cube
# input: the number of population
# output: basic 8 masses cube tree
def population(eva):
    newtree = []

    for t in range(eva):
        heap = []
        
        for i in range(1):
            exec('var{}={}'.format(i, {'pos':i,'children':list(range(1,7)),'depth':0,\
                                       'face':'default','cube':1,'spring_type':'default'}))
            exec('heap.append(var{})'.format(i))
        spring_type =  set_material()   
        for i in range(1,7):
            face_id = compute_baseface(i)
            exec('var{}={}'.format(i, {'pos':i,'children':list(range(2+5*i,7+5*i)),'depth':1,\
                                       'face':face_id,'cube':1,'spring_type':spring_type}))
            exec('heap.append(var{})'.format(i))
            
        newtree.append(heap[:])       
            
    return newtree


# add a new cube to the tree
def add_cube(individual):
    L = len(individual)
    d = individual[-1]['depth']
    cube = individual[-1]['cube']
    k = 0
    f = d
    if d == 1:
        i = random.randrange(1,7) # when depth is 1, add a cube on the basic cube
    else:
        s = 1
        for i in range(d):
            s += 6*pow(5,d-i-1)
        if L < s: # if this depth is not full
            p = random.random()
            if p < 0.5: # according to the prob, add a new cube in former depth
                while(1):
                    i = compute_facerange(f-1)
                    if i >= L:
                        f = f-1
                    else:
                        if len(individual[i]['face'])<7:
                            break
            else:   # otherwise, add the new cube in the current depth
                pos = individual[-1]['pos']
                i = random.randrange(pos-4,pos+1)   # i is the position to insert cube which is not exist in tree
                if i >= L:
                    while(1):
                        k += 1  # compute k to decide the actual position L-k in tree to insert cube
                        if individual[L-k]['pos'] == i:
                            break;
        else:       # if former depth is full, add a new cube in current depth
            i = compute_facerange(d)

    # change the original face from 4 masses to 8 masses
    if i >= L:
        face_id, face2 = change_face(L-k,individual)
    else:
        face_id, face2 = change_face(i,individual)
        
    spring_type =  set_material()
    if f == d:
        cube = cube + 1
        exec('var{}={}'.format(i, {'pos':i,'children':list(range(2+5*i,7+5*i)),'depth':d,\
                                   'face':face_id,'cube':[cube-1,cube],'spring_type':spring_type}))
    else:
        cube = cube + 1
        exec('var{}={}'.format(i, {'pos':i,'children':list(range(2+5*i,7+5*i)),'depth':f,\
                                   'face':face_id,'cube':[cube-1,cube],'spring_type':spring_type}))
   
    # add five new faces
    if i >= L:
        exec('individual[L-k] =(var{})'.format(i))
    else:
        
        exec('individual[i] =(var{})'.format(i))

    face_id = compute_face(face2,individual)    
    for j in range(2+5*i,7+5*i):
        if f == d:
            exec('var{}={}'.format(j, {'pos':j,'children':list(range(2+5*j,7+5*j)),'depth':d+1,\
                                        'face':face_id[j-(2+5*i)],'cube':cube,'spring_type':spring_type}))
        else:
            exec('var{}={}'.format(j, {'pos':j,'children':list(range(2+5*j,7+5*j)),'depth':f+1,\
                                        'face':face_id[j-(2+5*i)],'cube':cube,'spring_type':spring_type}))
        exec('individual.append(var{})'.format(j))

    return individual

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

def is_same_point(pos0, pos1):
    flag0 = np.abs(pos0[0] - pos1[0]) < 0.0001
    flag1 = np.abs(pos0[1] - pos1[1]) < 0.0001
    flag2 = np.abs(pos0[2] - pos1[2]) < 0.0001

    return flag0 and flag1 and flag2

def get_cube(cube, masses, springs, info, spring_type):
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
        p[0] = round(p[0],4)  #remove the floating error
        p[1] = round(p[1],4)
        p[2] = round(p[2],4)

    if len(info.id_list) > 0:
        max_id = np.max(info.id_list)
    else:
        max_id = -1
    this_cube = []  # mass list of this new cube (can include old mass point)
    if info.point_list:  # if it is not the first cube
        for i in range(len(pos)):  # for every pos[i], we have to know if it is already in the point_list
            if pos[i] in info.point_list: # find the index of the exist mass point in the mass list
                this_cube.append(masses[info.point_list.index(pos[i])]) # get this mass in this_cube, but not masses,
                                                                            # because we already have it!
            else:  # if this pos[i] is not in the point_list at all, we just add it as new mass point
                new_id = max_id + 1 + i
                new_mass = mass(m, pos[i], [0, 0, 0], [0, 0, 0], new_id)

                masses.append(new_mass)  # add this mass in this_cube, also in masses
                this_cube.append(new_mass)

                info.id_list.append(new_id)
                info.point_list.append(pos[i])

    else:           # create first cube
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
                    k, a, b = get_material(spring_type)
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
    fitness = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    # bouncing fitness
    # fitness = np.abs(z1-z0)

    return fitness
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

def compute_center(individual,h):
    robot_center =[]
    cube_num = individual[-1]['cube']
    count = 1
    for i in range(cube_num):
        if i == 0:
            robot_center.append([-1,h,0.3])
        else:
            robot_center.append([0,0,0])
    flag = 0
    pos = []
    L = len(individual)
    while count < cube_num:
        for i in range(1,L):
            if len(individual[i]['face'])>6:
                cube = individual[i]['cube']
                if cube[0] == count:
                    pos = np.array(robot_center[cube[0]-1])
                    face_id = individual[i]['face']
                    face = face_id.split(' ', 1 )
                    
                    face_list = ['top','bot','front','back','left','right']
                    vec = np.array([[0,0,0.1],[0,0,-0.1],[0.1,0,0],[-0.1,0,0],[0,-0.1,0],[0,0.1,0]])    
                    
                    for j in range(6):
                        if face[0] == face_list[j]:
                            flag  = j
        
                    robot_center[cube[1]-1] = (list(pos + vec[flag]))     
                    flag = 0
                    count = count+1
    cube = robot_center
    h2 = []
    for i in range(len(cube)):
            h2.append(cube[i][2])
         
    h = min(h2)
    for i in range(len(cube)):
            cube[i][2] = cube[i][2]-h+0.05
    return robot_center

def vis_run_model(robot):
    T = 0
    ts = 2000
    #robot =  [masses_v, springs_v, spheres_v, sides_v,info_v]
    masses = robot[0]
    springs = robot[1]
    mass_idList = robot[2]
    spheres = robot[3]
    sides = robot[4]

    mass_count = len(masses)

    for i in range(ts):
        rate(10000)
        # check each mass for its force
        for j in range(mass_count):
            # set up gravity
            force = 30*masses[j].m * deepcopy(g)

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

        T += dt
        print('time_step', i)

def compute_fitness(robot):
    T = 0
    ts = 4000
    #robot =  [masses_v, springs_v, spheres_v, sides_v,info_v]
    masses = robot[0]
    springs = robot[1]
    mass_idList = robot[2]
    spheres = robot[3]

    mass_count = len(masses)

    for i in range(ts):

        # check each mass for its force
        for j in range(mass_count):
            # set up gravity
            force = 20*masses[j].m * deepcopy(g)

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
            
        center = getcenter(masses)     
        fitness = center[0]   
        T += dt
    
        return fitness

canvas(width=1200, height=600, center=vector(0, 0, 0), background=color.white)
grid_ground(2, 25)

eva = 10
# cube_number = 7
pop = population(eva)
masses1 = []; spheres1=[]; springs1=[]; sides1=[]; info1 =[]
for j in range(eva):
    cube_number = random.randint(4, 6)
    spring_type=[]
    masses = []
    springs = []
    info = infomation([],[],[],[],0.1)
    point_list0 = []
    connect_list0 = []
    id_list = []
    
    individual = pop[j]
    for i in range(cube_number):
        individual = add_cube(individual)
    cube = compute_center(individual,-1.5+1*j)
    
    for i in range(len(cube)):
        spring_type.append(set_material())
        
    for i in range(len(cube)):
        masses, springs, info = get_cube(cube[i], masses, springs, info, spring_type[i])
        # spheres, sides = vis_cube(masses, info)
        
    masses1.append(masses); springs1.append(springs); info1.append(info)
    # masses1.append(masses); spheres1.append(spheres)
    # springs1.append(springs); sides1.append(sides)

spheres1 =[]
sides1 = []
T = 0
L = len(masses1)
mass_idList = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
               [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
               [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
               [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
               [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],]

for i in range(L):
    spheres, sides = vis_cube(masses1[i], info1[i])
    for ms in masses1[i]:
        mass_idList[i].append(ms.id)
    spheres1.append(spheres)
    sides1.append(sides)
    del(mass_idList[i][0])

robot =[]      

for i in range(eva):
    r1 = [masses1[i], springs1[i], mass_idList[i], spheres1[i], sides1[i]]
    robot.append(r1)  

Parallel(n_jobs=8, backend="threading")(
              map(delayed(vis_run_model),robot))  