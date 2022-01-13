
from vpython import *
import random
import numpy as np


class mass:
    def __init__(self, m, p, v, a, ID):
        self.m = m  # mass
        self.p = p  # position
        self.v = v  # velocity
        self.a = a  # acceleration
        self.id = ID

class spring:
    def __init__(self, k, l, c, a, b):
        self.k = k  # spring constant
        self.l = l  # rest length
        self.c = c  # indices of two connected masses by this spring
        self.a = a  # amplify factor for the expression a * sin(wt + b)
        self.b = b  # offset factor for the expression a * sin(wt + b)

def grid_ground(xmax, num):
    # xmax = extent of grid in each direction
    # num = number of lines

    # create vertical lines:
    l = np.linspace(-xmax, xmax, num=num)
    for i in l:
        curve(pos=[vector(i, xmax, 0), vector(i, -xmax, 0)])
        curve(pos=[vector(xmax, i, 0), vector(-xmax, i, 0)])

# compute the base six faces mass id
def compute_baseface(i):
    mass_id = []
    if i == 0:
        mass_id = list(range(8))
    if i > 0:
        mass_list = [[0,1,2,3],[0,1,4,5],[0,2,4,6],[1,3,5,7],[2,3,6,7],[4,5,6,7]]
        mass_id = mass_list[i-1]
    return mass_id

# connect the original face to other four mass points, comupte their mass id
# input: insert cube position, individual tree 
# output: changed face new mass id
def change_face(i,individual):
    L = len(individual)
    mass_id = []
    mass_id = individual[i]['massid']
    for j in range(4):
        mass_id.append(int(4*(L-7)/5+8+j)) # add 4 masses to the number of orignal robot L
    return mass_id

# compute the added five faces mass id
# input: insert cube position; new five faces range; original tree length;
#        the actual position to insert cube in tree L-k; individual tree
# output: new five faces mass id
def compute_face(i,j,L,k,individual):
    j = j - 2-5*i
    mass_id = []
    s = [[0,1],[0,2],[1,3],[2,3]]
    if k == 0:
        massid = individual[i]['massid']
    else:
        massid = individual[L-k]['massid']
    if j < 4:                                   # the first-fourth face
        mass_id.append(massid[s[j][0]])         
        mass_id.append(massid[s[j][1]])
        mass_id.append(massid[s[j][0]+4])
        mass_id.append(massid[s[j][1]+4])
    else:                                       # then fifth face
        for e in range(4):
            mass_id.append(massid[e-4])
    return mass_id

# when we try to add a cube, compute what pos we could add a cube, accroding to the depth
# return the position where insert a cube
def compute_massrange(d):
    s = 1
    for i in range(d):
        s += 6*pow(5,d-i-1)
    i = random.randrange(s-6*pow(5,d-1),s) 
    return i

# generate the population of the basic 8 masses cube
# input: the number of population
# output: basic 8 masses cube tree
def population(eva):
    newtree = []

    for t in range(eva):
        heap = []
        
        for i in range(1):
            mass_id = compute_baseface(i)
            exec('var{}={}'.format(i, {'pos':i,'children':list(range(1,7)),'depth':0,\
                                       'massid':mass_id,'springs':'default'}))
            exec('heap.append(var{})'.format(i))
        
        for i in range(1,7):
            mass_id = compute_baseface(i)
            exec('var{}={}'.format(i, {'pos':i,'children':list(range(2+5*i,7+5*i)),'depth':1,\
                                       'massid':mass_id,'springs':'default'}))
            exec('heap.append(var{})'.format(i))
            
        newtree.append(heap[:])       
            
    return newtree

# add a new cube to the tree
def add_cube(individual):
    L = len(individual)
    d = individual[-1]['depth']
    k = 0
    f = d
    if d < 2:
        i = random.randrange(1,7) # when depth is 1, add a cube on the basic cube
    else:
        s = 1
        for i in range(d):
            s += 6*pow(5,d-i-1)
        if L < s: # if this depth is not full
            p = random.random()
            if p < 0.5: # according to the prob, add a new cube in former depth
                while(1):
                    i = compute_massrange(f-1)
                    if i >= L:
                        f = f-1
                    else:
                        if len(individual[i]['massid'])<5:
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
            i = compute_massrange(d)

    # change the original face from 4 masses to 8 masses
    if i >= L:
        mass_id = change_face(L-k,individual)
    else:
        mass_id = change_face(i,individual)
    
    if f == d:
        exec('var{}={}'.format(i, {'pos':i,'children':list(range(2+5*i,7+5*i)),'depth':d,\
                                   'massid':mass_id,'springs':'default'}))
    else:
        exec('var{}={}'.format(i, {'pos':i,'children':list(range(2+5*i,7+5*i)),'depth':f,\
                                   'massid':mass_id,'springs':'default'}))
   
    # add five new faces
    if i >= L:
        exec('individual[L-k] =(var{})'.format(i))
    else:
        exec('individual[i] =(var{})'.format(i))

    for j in range(2+5*i,7+5*i):
        mass_id = compute_face(i,j,L,k,individual)
        if f == d:
            exec('var{}={}'.format(j, {'pos':j,'children':list(range(2+5*j,7+5*j)),'depth':d+1,\
                                        'massid':mass_id,'springs':'default'}))
        else:
            exec('var{}={}'.format(j, {'pos':j,'children':list(range(2+5*j,7+5*j)),'depth':f+1,\
                                        'massid':mass_id,'springs':'default'}))
        exec('individual.append(var{})'.format(j))

    return individual

def generate_cube(i,individual,masses,spheres,r1):
    flag = 0
    pos = []
    mass_id = individual[i]['massid']
    
    for j in range(4):
        md = mass_id[j]
        pos.append(np.array(masses[md].p))
    vec1 = pos[1] - pos[0]
    vec2 = pos[2] - pos[0]
    vec = np.abs(np.cross(vec1,vec2))

    if vec[0] != 0:
        vec[0] = 0.1
    elif vec[1] != 0:
        vec[1] = 0.1
    elif vec[2] != 0:
        vec[2] = 0.1

    test_vec = list(pos[0] - vec) 

    for j in range(len(masses)):
        if masses[j].p == test_vec:
            flag = 1        
    
    for j in range(4):
        if flag == 1:
            pos[j] = list(pos[j] + vec)
        else:
            pos[j] = list(pos[j] - vec)
        masses.append(mass(0.1, pos[j], [0, 0, 0], [0, 0, 0], mass_id[4+j]))  # cube corners
        spheres.append(sphere(pos=vector(masses[-1].p[0], masses[-1].p[1],
                                              masses[-1].p[2]), radius=r1, color=color.red))
    return masses,spheres


def generate_springs(springs,sides,masses,spheres,individual,r2):
    k = 0
    a = 0.1
    b = 0
    L = len(individual)
    for k in range(L):
        if len(individual[k]['massid'])>4:
            mass_id = individual[k]['massid']
            for i in mass_id:
                for j in mass_id:
                    if i < j:    
                        initial_vec =[  masses[j].p[0]-masses[i].p[0],
                                    masses[j].p[1]-masses[i].p[1],
                                    masses[j].p[2]-masses[i].p[2]]  
                        l = np.linalg.norm(initial_vec)
                        
                        springs.append(spring(k, l, [i, j], a, b))
                        sides.append(
                            cylinder(pos=spheres[i].pos,
                                      axis=vector(initial_vec[0], initial_vec[1],
                                                  initial_vec[2]),
                                      color=color.blue, radius=r2))
    return springs, sides

def generate_robot(individual,cube_number):
    h = 0.4
    m = 0.1
    r1 = 0.01  # radius of the ball
    r2 = 0.005  # radius of the cylinder
    L = len(individual) 
    count = 4
    masses = []  # a list of mass
    spheres = []  # a list of spheres

    initial_pos = [[0, 0, h], [0.1, 0, h], [0, 0.1, h], [0.1, 0.1, h],
                        [0, 0, h + 0.1], [0.1, 0, h + 0.1], [0, 0.1, h + 0.1],
                        [0.1, 0.1, h + 0.1]]

    for i, pos_item in enumerate(initial_pos):
        masses.append(mass(m, pos_item, [0, 0, 0], [0, 0, 0], i))  # cube corners
        spheres.append(sphere(pos=vector(masses[i].p[0], masses[i].p[1],
                                          masses[i].p[2]), radius=r1, color=color.red))
    while count < 4+4*cube_number:
        for i in range(1,L):
            if len(individual[i]['massid'])>4:
                massid = individual[i]['massid'] 
                if massid[4] == 4+count:
                    count = count+4
                    masses,spheres = generate_cube(i,individual,masses,spheres,r1)
        
    springs = []  # a list of spring object
    sides = []  # a list of cylinder
    
    springs,sides = generate_springs(springs,sides,masses,spheres,individual,r2)

    return masses,spheres,springs,sides   

canvas(width=1200, height=600, center=vector(0, 0, 0), background=color.white)
grid_ground(1, 20)
eva =1

cube_number = 6
pop = population(eva)
individual = pop[0]
for i in range(cube_number):
    individual = add_cube(individual)

masses,spheres,springs,sides  = generate_robot(individual,cube_number)


