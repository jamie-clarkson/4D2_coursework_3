# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:10:16 2022

@author: Jamie
"""

""" Import required libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import ndimage, misc
%matplotlib inline
import meshzoo

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cmap1 = LinearSegmentedColormap.from_list('mycmap', ['white', 'red'],gamma=2)
cmap2 = LinearSegmentedColormap.from_list('mycmap', ['white', 'green'],gamma=2)


"""Define roof dimensions"""
global L_width_x, L_width_y, N_nodes_x, N_nodes_y
#roughly 120m by 60m, (y by x)
L_width_x = 63
L_width_y = 120
N_nodes_x = 21
N_nodes_y = 41 #y direction should be odd number of nodes  #should probably double in y direction

"""Calculate nodal loading """
#loading in kpa, say 1 for wind and 1 for self weight so 2kpa
#work out tributary area for node
A = (L_width_x/N_nodes_x)*(L_width_y/N_nodes_y)
w = 2 #kpa
global node_load
node_load = A*w


"""This bit does the meshing """
points, cells = meshzoo.rectangle_tri(
    np.linspace(-L_width_x/2, L_width_x/2, N_nodes_x),
    np.linspace(-L_width_y/2, L_width_y/2, N_nodes_y),
    variant="zigzag",  # or "up", "down", "center"
)

points_original = points[:]
""" Need to modify points to create arched grid"""
points_mod = []
new_arch = []

left_arch = []
for point in points:
    x = point[0]
    y = point[1]

    if (x == L_width_x/2):
        new_arch.append(1)
    if (x != L_width_x/2):
        new_arch.append(0)
        
    if (x == -L_width_x/2 + L_width_x/(N_nodes_x-1)):
        left_arch.append(1)
    if (x != -L_width_x/2 + L_width_x/(N_nodes_x-1)):
        left_arch.append(0)
        
    if (x != -L_width_x/2) and (y != -L_width_y/2) and (y != L_width_y/2):
        scalex = (1/1500)*(x**2)/((L_width_x/2)**2)   #more arch seems to help
        scaley = (1/200)*(y**2)/((L_width_y/2)**2)   
        x = x + np.sign(x)*((y**2)*scalex - ((L_width_y/2)**2)*scalex)
        y = y + np.sign(y)*((x**2)*scaley - ((L_width_x/2)**2)*scaley)
    points_mod.append([x,y])
        

    
    
points_mod = np.array(points_mod)
points = points_mod

points_2d = points[:]
points1 = points   # need to add z coords#
points = np.insert(points, 2, 0,1)   #adding z coords








""" Obtaining list of edges in the grid"""
new_arch_bars = []
left_arch_bars = []
#need a way to find out what other points each point is itself connected to...
edges_a = [[]]*len(points)
edges_b = []
for cell in cells:
    edge_0 = [cell[0],cell[1]]
    edge_1 = [cell[1],cell[2]]
    edge_2 = [cell[2],cell[0]]
    these_edges = [edge_0,edge_1,edge_2]
    for e in these_edges:
        if e not in edges_b and e.reverse() not in edges_b:
            edges_b.append(e)
            if new_arch[e[0]] == 1 and new_arch[e[1]] == 1:
                new_arch_bars.append(1)
            else:
                new_arch_bars.append(0)
                
            if left_arch[e[0]] == 1 and left_arch[e[1]] == 1:
                left_arch_bars.append(1)
            else:
                left_arch_bars.append(0)
                
edges_b.append([N_nodes_x - 1,N_nodes_x*N_nodes_y -1])
""" Need to add the arch cell"""
new_cell = [] 
i = 0
for p in points_original:
    x = p[0]
    if x == L_width_x/2:
        new_cell.append(i)        
    i+=1
 
cells_all = cells[:]      
cells_all = list(cells_all) 
cells_all.append(new_cell)

new_arch_bars.append(0)
left_arch_bars.append(0)

edges_b = np.array(edges_b)
for edge in edges_b:
    if edge[0] == edge[1]:
        print('edge error')
connections_to_other_points_ind = [[]]*len(points)

connections_to_other_points = [[]]*len(points)

global valid_finders
valid_finders = [[]]*len(points)

df = pd.DataFrame(edges_b, columns = ['point_1','point_2'])

for i in range(0,len(points)):
    #point is i
    point = i
    these_edges = list(df.loc[(df['point_1']==point)]['point_2']) + list(df.loc[(df['point_2']==point)]['point_1'])
    these_edges2 = [points[k] for k in these_edges]
    valid_finder = [0]*len(these_edges) 
    for k in range(0,8-len(these_edges)):
        these_edges.append(1)
        valid_finder.append(np.nan)
        these_edges2.append(np.array([np.nan,np.nan,np.nan]))
    connections_to_other_points_ind[i] = these_edges 
    connections_to_other_points[i] = these_edges2
    valid_finders[i] = valid_finder
    
    

connections_to_other_points = np.rollaxis(np.transpose(np.dstack(np.array(connections_to_other_points))),1)
connections_to_other_points_ind = np.array(connections_to_other_points_ind)   #how to get from this to that given points
valid_finders = np.array(valid_finders)   #how to get from this to that given points

connections_to_other_points_ind=connections_to_other_points_ind.astype(int)



""" Need a way to find the 90 degree rotational symmetry"""


""" First find the centre point of each bar (iterate through bars)
From this construct a dictionary, where the key is the x y and the item is the bar number

Then to do a rotation we can rotate the x y coords, use the dictionary to find which bar it is
(Only works for symmetric meshes)
 """

def find_bar_midpoints(bars,points):
    mid_points = []
    bar_dict = {}
    i = 0
    for bar in bars:
        p1 = np.array(points[bar[0]])
        p2 = np.array(points[bar[1]])
        #print(p1,p2)
        mid = (p1+p2)/2
        mid_points.append(mid)
        tol = 2
        key = (round(mid[0],tol),round(mid[1],tol))
        bar_dict[key] = i
        i += 1
    return mid_points, bar_dict
mid_points, bar_dict = find_bar_midpoints(edges_b,points_2d)

def calc_rotation_matrix(rotation_in_degrees):
    theta = np.radians(rotation_in_degrees)
    rotation_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return rotation_matrix

def rotate_point(point,rotation_matrix):
    point = np.array(point)
    rotated_point = np.matmul(rotation_matrix,point) 
    return rotated_point

#try a 90 degree rotation


def rotate_all_points(points,rotation_in_degrees):
    rotation_matrix = calc_rotation_matrix(rotation_in_degrees)
    rotated_points = []
    for point in points:
        rotated_point = rotate_point(point, rotation_matrix)
        rotated_points.append(rotated_point)
    return rotated_points

#rotated_points = rotate_all_points(points_2d, 23)

def transform_stress_state(mid_points,stress_state,quarter_rotations,bar_dict):
    rotated_points = rotate_all_points(mid_points, quarter_rotations*90)
    # for each point in rotate points, find the old stress
    rotated_stress_state = []
    for bar_mid_point in rotated_points:
        tol = 2
        key = (round(bar_mid_point[0],tol),round(bar_mid_point[1],tol))
        old_bar_index = bar_dict[key]
        old_stress = stress_state[old_bar_index]
        rotated_stress_state.append(old_stress)
    return rotated_stress_state

""" Check if self stress state is valid"""
def is_self_stress_state(equilibrium_matrix,stress_state):
    return np.matmul(equilibrium_matrix,stress_state)

""" Create rotationally symmetric state of self stress by summing rotations"""
def symmetrify_stress(mid_points,stress_state,bar_dict):
    original_stress = stress_state
    for i in range(1,4):
        #print(i)
        rotated_stress = np.array(transform_stress_state(mid_points,stress_state,i,bar_dict))  
        original_stress = original_stress + rotated_stress
    return original_stress

""" Create valid state of self stress by projecting guess onto nullspace"""
def project_stress(basis_vectors,target):
    basis_vectors = np.transpose(basis_vectors)
    target = np.array(target)
    output = np.zeros(len(target))
    for basis_vector in basis_vectors:
        output = output + basis_vector*np.dot(basis_vector,target)
    return output

""" Find the outer edges of the grid"""
def find_outer_edges(mid_points,x_limits,y_limits):
    mid_points_array = np.transpose(np.array(mid_points))
    
    x0 = min(mid_points_array[0])
    x1 = max(mid_points_array[0])
    y0 = min(mid_points_array[1])
    y1 = max(mid_points_array[1])
    tol = 3
    x_limits = (round(x0,tol),round(x1,tol))
    y_limits = (round(y0,tol),round(y1,tol))
    is_outer = []
    for bar in mid_points:
        bx = round(bar[0],tol)
        by = round(bar[1],tol)
        if bx in x_limits or by in y_limits:
            is_outer.append(1)
        else:
            is_outer.append(0)
    return is_outer


def find_outer_edges_2(mid_points,x_limits,y_limits):
    mid_points_array = np.transpose(np.array(mid_points))
    
    x0 = min(mid_points_array[0])
    x1 = max(mid_points_array[0])
    y0 = min(mid_points_array[1])
    y1 = max(mid_points_array[1])
    tol = 3
    x_limits = (round(x0,tol),round(x1,tol))
    y_limits = (round(y0,tol),round(y1,tol))
    is_outer = []
    for bar in mid_points:
        bx = round(bar[0],tol)
        by = round(bar[1],tol)
        if bx == x0 or by in y_limits:
            is_outer.append(1)
        else:
            is_outer.append(0)
    return is_outer
""" Find a specific outer edge of the grid"""
def find_specific_edge(points,desired_edge):
    points_array = np.transpose(np.array(points))
    
    x0 = min(points_array[0])
    x1 = max(points_array[0])
    y0 = min(points_array[1])
    y1 = max(points_array[1])
    tol = 3  
    
    
    is_outer = []
    for bar in points:
        bx = round(bar[0],tol)
        by = round(bar[1],tol)
        if desired_edge == 'left':
            if bx == x0:
                is_outer.append(1)
            else:
                is_outer.append(0)
        if desired_edge == 'right':
            if bx == x1:
                is_outer.append(1)
            else:
                is_outer.append(0)
        if desired_edge == 'top':
            if by == y1:
                is_outer.append(1)
            else:
                is_outer.append(0)
        if desired_edge == 'bottom':
            if by == y0:
                is_outer.append(1)
            else:
                is_outer.append(0)
                                                                
    return is_outer




def find_outer_edge_3(mid_points,x_limits,y_limits):
    mid_points_array = np.transpose(np.array(mid_points))
    
    x0 = min(mid_points_array[0])
    x1 = max(mid_points_array[0])
    y0 = min(mid_points_array[1])
    y1 = max(mid_points_array[1])
    tol = 3
    x_limits = (round(x0,tol),round(x1,tol))
    y_limits = (round(y0,tol),round(y1,tol))
    is_outer = []
    for bar in mid_points:
        bx = round(bar[0],tol)
        by = round(bar[1],tol)
        if bx == x1:
            is_outer.append(1)
        else:
            is_outer.append(0)
    return is_outer


outer_bars = find_outer_edges(mid_points,(L_width_x/2, L_width_x/2),(L_width_x/2, L_width_x/2))    
outer_bars_count = np.count_nonzero(outer_bars)    

""" Find the nodes on the boundaries"""
def find_outer_nodes(points,x_limits,y_limits):
    points_array = np.transpose(np.array(points))
    
    x0 = min(points_array[0])
    x1 = max(points_array[0])
    y0 = min(points_array[1])
    y1 = max(points_array[1])
    tol = 3
    x_limits = (round(x0,tol),round(x1,tol))
    y_limits = (round(y0,tol),round(y1,tol))
    is_outer = []
    for node in points:
        bx = round(node[0],tol)
        by = round(node[1],tol)
        if bx in x_limits or by in y_limits:
            is_outer.append(1)
        else:
            is_outer.append(0)
    return is_outer

outer_nodes = find_outer_nodes(points,(L_width_x/2, L_width_x/2),(L_width_x/2, L_width_x/2))

""" Plot the gridshell"""
def plot_views(mesh_points,mesh_edges,option):  
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')
    
    if option == 1:
        ax.view_init(elev=0, azim=-90)
    if option == 2:
        ax.view_init(elev=0, azim=0)
    if option == 3:
        ax.view_init(elev=90, azim=0)
    if option == 4:
        ax.view_init(elev=0, azim=45)
    if option == 5:
        ax.view_init(elev=0, azim=-45)
    

    for edge in mesh_edges:
        p1 = np.transpose(mesh_points)[edge[0]]
        p2 = np.transpose(mesh_points)[edge[1]]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color = 'k')


    x_range = max(ax.get_xlim()) - min(ax.get_xlim())
    y_range = max(ax.get_ylim()) - min(ax.get_ylim())
    z_range = max(ax.get_zlim()) - min(ax.get_zlim())
    ax.set_box_aspect([x_range,y_range,z_range])
    
    plt.show()
    
""" Plot the gridshell, coloured by bar stress"""
def plot_views_3D_stress(mesh_points,mesh_edges,stresses,option):  
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')
    
    
    cmax = max(stresses)
    cmin = abs(min(stresses))
    
    print('cmax',cmax)
    print('cmin',cmin)
    
    cmax = max(cmax,cmin)
    
    if option == 1:
        ax.view_init(elev=0, azim=-90)
    if option == 2:
        ax.view_init(elev=0, azim=0)
    if option == 3:
        ax.view_init(elev=90, azim=0)
    if option == 4:
        ax.view_init(elev=0, azim=45)
    if option == 5:
        ax.view_init(elev=0, azim=-45)
    

    i = 0
    for edge in mesh_edges:
        p1 = np.transpose(mesh_points)[edge[0]]
        p2 = np.transpose(mesh_points)[edge[1]]
        
        bar_stress = stresses[i]  #this isn't workin
        i+=1
        if bar_stress < 0:
            col = (0,0,1,abs(bar_stress/cmax)) #blue compression
        if bar_stress >= 0:
            col = (1,0,0,abs(bar_stress/cmax)) #red tension
            
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color = col)


    x_range = max(ax.get_xlim()) - min(ax.get_xlim())
    y_range = max(ax.get_ylim()) - min(ax.get_ylim())
    z_range = max(ax.get_zlim()) - min(ax.get_zlim())

    ax.set_box_aspect([x_range,y_range,z_range])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    plt.savefig('design2_view' + str(option) +'.png',dpi=300)
    
    plt.show()    
    
""" Plot the gridshell showing the out of balance forces (a.k.a the reaction forces)"""
def plot_views_3D_out_of_balance_force(mesh_points,mesh_edges,out_of_balance_force,option):  
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')
    
    out_of_balance_force = np.transpose(out_of_balance_force)[2]
    print('out_of_balance_force',out_of_balance_force)
    cmax = max(out_of_balance_force)
    cmin = abs(min(out_of_balance_force))
    cmax = max(cmax,cmin)
    
    #out_of_balance_force = np.transpose(out_of_balance_force)
    print('max out of balance force',cmax)
    if option == 1:
        ax.view_init(elev=0, azim=-90)
    if option == 2:
        ax.view_init(elev=0, azim=0)
    if option == 3:
        ax.view_init(elev=90, azim=0)
    if option == 4:
        ax.view_init(elev=0, azim=45)
    if option == 5:
        ax.view_init(elev=0, azim=-45)
    

    i = 0
    for edge in mesh_edges:
        p1 = np.transpose(mesh_points)[edge[0]]
        p2 = np.transpose(mesh_points)[edge[1]]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color = (0,0,0,0.3))

    i = 0
    scale_factor = (1/cmax)*5
    mesh_points = np.transpose(mesh_points)
    for point in mesh_points:
        force = out_of_balance_force[i]
        i+=1
        #print('force',force)
        xs = [point[0],point[0]]
        ys = [point[1],point[1]]
        zs = [point[2],point[2]-force*scale_factor]
        plt.plot(xs,ys,zs,color = (1,0,0,1))


    x_range = max(ax.get_xlim()) - min(ax.get_xlim())
    y_range = max(ax.get_ylim()) - min(ax.get_ylim())
    z_range = max(ax.get_zlim()) - min(ax.get_zlim())

    ax.set_box_aspect([x_range,y_range,z_range])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    plt.savefig('net_forces' + str(option) +'.png',dpi=300)
    
    plt.show()    

""" Plot the gridshell showing a rough guess at the buckled state"""
def plot_views_buckled(mesh_points,mesh_edges,option):  
    #apply buckle to mesh_points
    mesh_points_buckled = []
    for point in mesh_points:
        x = point[0]
        y = point[1]
        z = point[2]
        A = 5
        new_z = z + A*np.sin(np.pi*(x+L_width_x/2)/L_width_x)*np.sin(2*np.pi*(y+L_width_y/2)/L_width_y) 
        new_point = [x,y,new_z]
        mesh_points_buckled.append(new_point)
    mesh_points = np.array(mesh_points_buckled)
    mesh_points = np.transpose(mesh_points) 
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')

    if option == 1:
        ax.view_init(elev=0, azim=-90)
    if option == 2:
        ax.view_init(elev=0, azim=0)
    if option == 3:
        ax.view_init(elev=90, azim=0)
    if option == 4:
        ax.view_init(elev=0, azim=45)
    if option == 5:
        ax.view_init(elev=0, azim=-45)
    
    #ax.scatter3D(x,y,z)
    #ax.plot_surface(x,y,z)

    for edge in mesh_edges:
        p1 = np.transpose(mesh_points)[edge[0]]
        p2 = np.transpose(mesh_points)[edge[1]]
        col = (0,0,0,0.8)
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color = col)


    x_range = max(ax.get_xlim()) - min(ax.get_xlim())
    y_range = max(ax.get_ylim()) - min(ax.get_ylim())
    z_range = max(ax.get_zlim()) - min(ax.get_zlim())

    ax.set_box_aspect([x_range,y_range,z_range])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    plt.savefig('design2_buckled' + str(option) +'.png',dpi=300)
    
    plt.show()        
    
""" Plot the 2D gridshell"""
def plot_frame(points,bars,non_reaction_points):
    maxx = 0
    minx = 0
    maxy = 0
    miny = 0
    
    for i in range(0,len(points)):
        p = points[i]
        x = p[0]
        y = p[1]
        
        if x > maxx:
            maxx = x
        if x < minx:
            minx = x
        if y > maxy:
            maxy = y
        if y < miny:
            miny = y
    
    plt.xlim(minx-0.5*(maxx-minx),maxx+0.5*(maxx-minx))
    plt.ylim(miny-0.5*(maxy-miny),maxy+0.5*(maxy-miny))

    
    L = 0
    for i in range(0,len(bars)):
        bar = bars[i]
        a = points[bar[0]]
        b = points[bar[1]]
        L = L + np.linalg.norm(np.array(b)-np.array(a))
        
        mid = (np.array(a)+np.array(b))/2
        plt.scatter(a[0],a[1],color = 'b')
        plt.scatter(b[0],b[1],color = 'b')
        X = [a[0],b[0]]
        Y = [a[1],b[1]]
        plt.plot(X,Y,color = 'k')  
        plt.annotate(str(i), mid,bbox=dict(facecolor='w', edgecolor='k'))
    L = L/len(bars)  
    for i in range(0,len(non_reaction_points)):
        p = points[non_reaction_points[i]]
        d = 0.1*L
        xf = [p[0]+0.2*L,p[1]]
        X = [p[0],xf[0]]
        Y = [p[1],xf[1]]
        plt.arrow(p[0],p[1],d,0,fc="r", ec="r",head_width=0.05, head_length=0.1)  
        plt.annotate('F' + str(i*2), xf)
        
        yf = [p[0],p[1]+0.2*L]
        X = [p[0],yf[0]]
        Y = [p[1],yf[1]]
        plt.arrow(p[0],p[1],0,d,fc="r", ec="r",head_width=0.05, head_length=0.1) 
        plt.annotate('F' + str(i*2+1), yf)

def normalised_vector(v,bar=[22,2]):
    if np.linalg.norm(v) != 0:
        return np.array(v)/np.linalg.norm(v)
    else:
        print('zero length vector ',v)
        print('bar',bar)
        return np.array([0]*len(v)) #in case the edge goes to itself
    
import scipy 
from scipy.linalg import null_space

from sympy import Matrix

""" Calculated a 2D equilibrium matrix"""        
def EQ_matrix2D(points,bars,non_reaction_points):  #actually the edges we have are the bars
    
 
    plt.show()
    forces = [0]*(2*len(non_reaction_points))

    H = [[0]*len(bars)]*len(forces)

    eq = []
    for n in range(0,len(non_reaction_points)):
        fxi = 2*n
        index = non_reaction_points[n] #n is no of point, index is which of the points it is
        p = points[index]  #this is the point in all the points
        #cs = connections[index]
        #x force first
        s = []
        for i in range(0,len(bars)):
            bar = bars[i]
            if index in bar:
                if bar[0] == index:
                    b = [bar[1],bar[0]]
                if bar[1] == index:
                    b = [bar[0],bar[1]]
                
                x = p
                a = points[b[0]]
                f = np.array(x)-np.array(a) # this means x is the same position as a
                M = np.dot(np.array([1,0]),normalised_vector(f,bar))  #suspect f is nan, because point is connecting to itself?
                H[fxi][i] = M

                    
                s.append(M)
            if index not in bar:
                s.append(0)
            #print(H[1])
            #H[2][3] = 76
        eq.append(s)
        #now y force
        fyi = 2*n +1
        s = []
        for i in range(0,len(bars)):
            bar = bars[i]
            if index in bar:
                if bar[0] == index:
                    b = [bar[1],bar[0]]
                if bar[1] == index:
                    b = [bar[0],bar[1]]
                x = p
                a = points[b[0]]
                f = np.array(x)-np.array(a)
                M = np.dot(np.array([0,1]),normalised_vector(f,bar))
 
                  
                H[fyi][i] = M
                #print(np.array(H))
                
                s.append(M)
            if index not in bar:
                s.append(0)
        #print(n, ' ',s)
        eq.append(s)
    H = np.array(H)
    #print(H)
    global equilibrium_matrix, nan_count
    eq = np.array(eq)
    equilibrium_matrix = eq
    #find number of nans
    nan_count = np.count_nonzero(np.isnan(equilibrium_matrix))  #we have four nans
    print(eq)    
    print(eq.shape)
    #print(np.linalg.matrix_rank(eq))    


    C = np.transpose(eq)
    
    self_stress_states = null_space(eq)
    mechanisms = null_space(C)
    print('States of self stress: ', self_stress_states)
    print('Mechanisms: ', mechanisms)
    
    mechs = np.size(mechanisms,1)
    stresses = np.size(self_stress_states,1)
    
    print(str(mechs) + ' mechanisms and ' + str(stresses) + ' states of self stress')

    return eq,mechanisms,self_stress_states

non_reaction_points = []  #this needs to be all the non reaction points!! If left empty, there will be three mechanisms (x,y translation plus rotation)
#rection_points = [N_nodes_x - 1,N_nodes_x*N_nodes_y -1]
rection_points = []
for i in range(0,len(points_2d)):
    if i not in rection_points:
        non_reaction_points.append(i)
print('here')
eq,mechanisms,self_stress_states = EQ_matrix2D(points_2d,edges_b,non_reaction_points)

#self_stress_states = np.load('self_stress_states_21.npy')

print('equilibrium calculated')

""" Plot the 2D gridshell, bars coloured by stress"""
def plot_frame2D_selfstress(points,bars,non_reaction_points,self_stress_states,N_stress,guess = False):
    self_stress_states = np.transpose(self_stress_states)

    maxx = 0
    minx = 0
    maxy = 0
    miny = 0

    
    for i in range(0,len(points)):
        p = points[i]

        x = p[0]
        y = p[1]
        
        if x > maxx:
            maxx = x
        if x < minx:
            minx = x
        if y > maxy:
            maxy = y
        if y < miny:
            miny = y


    L = 0
    
    for s in range(0,N_stress):
        ax = plt.axes()
        ax.set_xlim(minx-0.5*(maxx-minx),maxx+0.5*(maxx-minx))
        ax.set_ylim(miny-0.5*(maxy-miny),maxy+0.5*(maxy-miny))
        if guess == False:
            plt.title('Self stress state ' + str(s+1))
        if guess == True:
            plt.title('Guessed stress state')            
        # print(m)
        stresses = self_stress_states[s]
        # print('mech: ', mech)
        cmax = max(stresses)
        cmin = abs(min(stresses))
        cmax = max(cmax,cmin)
        for i in range(0,len(bars)):
            bar_stress = stresses[i]
    
            if bar_stress < 0:
                col = (0,0,1,abs(bar_stress/cmax)) #blue compression
            if bar_stress >= 0:
                col = (1,0,0,abs(bar_stress/cmax)) #red tension
                
            bar = bars[i]
            a = points[bar[0]]
            b = points[bar[1]]
            L = L + np.linalg.norm(np.array(b)-np.array(a))
            
            mid = (np.array(a)+np.array(b))/2
            #ax.scatter(a[0],a[1],color = 'b')
            #ax.scatter(b[0],b[1],color = 'b')
            X = [a[0],b[0]]
            Y = [a[1],b[1]]
            ax.plot(X,Y,color = col)  
            #ax.text(mid[0],mid[1],mid[2],(i),fontsize = 8)
            
        ax = plt.gca()
        ax.set_aspect(1)
        ax.set_ylim(-70,70)
        
        plt.show()
        plt.close()


global points5
points5 = points
def get_point(index):
    return points5[index] 

get_point_vectorised = np.vectorize(get_point,excluded=['points'],signature='(m,n)->(m,n,3)')

global all_edges
all_edges = edges_b

""" Get the bar ids that each node is connected to"""
def get_connecting_bar_id(connections_to_other_points_ind,edges,valid_finder):
    edges = edges.tolist()
    global edge_list
    edge_list = edges
    bars = connections_to_other_points_ind*0
    i = 0
    output_bars = []
    connections_to_other_points_ind_list = connections_to_other_points_ind.tolist()
    for row in connections_to_other_points_ind_list:
        output_row = []
        j = 0
        #print('row',row)  #the row looks okay
        for item in row:
            #print('j',j)
            #print('item',item)
            node_id = int(i)
            connecting_node_id = int(item)
            edge = [node_id,connecting_node_id]
            edge2 = [connecting_node_id,node_id]
            global this_edge
            this_edge = edge
            #print(this_edge)
            if edge in edge_list:
                bar_id = edge_list.index(edge)
                #print(bar_id)   #so this does work, but for some reason they bcome in
                #bars[i][j] = bar_id
                output_row.append(bar_id)
                #print(i,j,'normal')
            elif edge2 in edge_list:
                bar_id = edge_list.index(edge2)
                #bars[i][j] = bar_id   
                output_row.append(bar_id)
                #print(i,j,'reverse')
            else:
                #bars[i][j] = 666 
                output_row.append(6666)
                #print(i,j,'else')
            j = j + 1  #problem with j
            
        output_bars.append(output_row)
        #print(output_bars)
        #f = input('halt')
        i += 1
    return np.array(output_bars)

""" Turn the 1D array of stresses into the 2D array used in the formfinding"""       
def get_original_stress_state(connections_to_other_points_ind,edges,original_stress_state,valid_finder):
    edges = edges.tolist()
    original_stress_state = original_stress_state.tolist()
    global edge_list
    edge_list = edges
    bars = connections_to_other_points_ind*0
    i = 0
    output_stress = []
    connections_to_other_points_ind_list = connections_to_other_points_ind.tolist()
    for row in connections_to_other_points_ind_list:
        output_row = []
        j = 0
        #print('row',row)  #the row looks okay
        for item in row:
            #print('j',j)
            #print('item',item)
            node_id = int(i)
            connecting_node_id = int(item)
            edge = [node_id,connecting_node_id]
            edge2 = [connecting_node_id,node_id]
            global this_edge
            this_edge = edge
            #print(this_edge)
            if edge in edge_list:
                bar_id = edge_list.index(edge)
                #print(bar_id)   #so this does work, but for some reason they bcome in
                #bars[i][j] = bar_id
                stress = original_stress_state[bar_id][0]
                output_row.append(stress)
                #print(i,j,'normal')
            elif edge2 in edge_list:
                bar_id = edge_list.index(edge2)
                #bars[i][j] = bar_id   
                stress = original_stress_state[bar_id][0]
                output_row.append(stress)
            else:
                #bars[i][j] = 666 
                output_row.append(0)
                #print(i,j,'else')
            j = j + 1  #problem with j
            
        output_stress.append(output_row)
        #print(output_bars)
        #f = input('halt')
        i += 1
    return np.array(output_stress)

get_bars_vectorised = np.vectorize(get_connecting_bar_id,excluded=['edges'],signature='(m,n)->(m,n)')

points5 = points
new_points_attempt = np.rollaxis(get_point_vectorised(connections_to_other_points_ind),2)+valid_finders
global new_bars_attempt
new_bars_attempt = get_connecting_bar_id(connections_to_other_points_ind,edges_b,valid_finder) +valid_finders  #this seems to work

"""Another plotter""" 
def plot_mesh(mesh_points,mesh_edges):
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')
    
    for edge in mesh_edges:
        global e, p1, p2
        e = edge
        p1 = np.transpose(mesh_points)[edge[0]]
        p2 = np.transpose(mesh_points)[edge[1]]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color = 'k')

    ax.scatter3D(x,y,z,color='b')
    plt.show()



"""Another plotter""" 
def plot_views(mesh_points,mesh_edges,option):  
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')
    
    if option == 1:
        ax.view_init(elev=0, azim=-90)
    if option == 2:
        ax.view_init(elev=0, azim=0)
    if option == 3:
        ax.view_init(elev=90, azim=0)
    if option == 4:
        ax.view_init(elev=0, azim=45)
    if option == 5:
        ax.view_init(elev=0, azim=-45)
    
    #ax.scatter3D(x,y,z)
    #ax.plot_surface(x,y,z)
    
    for edge in mesh_edges:
        p1 = np.transpose(mesh_points)[edge[0]]
        p2 = np.transpose(mesh_points)[edge[1]]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color = 'k')


    x_range = max(ax.get_xlim()) - min(ax.get_xlim())
    y_range = max(ax.get_ylim()) - min(ax.get_ylim())
    z_range = max(ax.get_zlim()) - min(ax.get_zlim())
    ax.set_box_aspect([x_range,y_range,z_range])
    
    plt.show()





"""Another plotter - doesn't draw the edges to save time""" 
def plot_views_quick(mesh_points):
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    ax = plt.axes(projection='3d')

    ax.scatter3D(x,y,z,color='b')

"""Plot how out of balance a state of stress is""" 
def plot_views_quick_out_of_balance(mesh_points,eq_results):
    eq_results = np.transpose(eq_results)[0]
    print(eq_results)
    xs = []
    ys = []
    for i in range(0,len(eq_results)):
        f = eq_results[i]
        if i%2 == 0:
            xs.append(f)
        else:
            ys.append(f)
    xs = np.array(xs)
    ys = np.array(ys)
    tots = (xs**2 + ys**2)**0.5
    max_tot = max(tots)
    mesh_points = np.transpose(mesh_points)
    x = mesh_points[0]
    y = mesh_points[1]
    z = mesh_points[2]
    fig= plt.figure(figsize=(6,6)) # set figure size to portrait
    #ax = plt.axes(projection='3d')
    #ax.view_init(elev=90, azim=0)
    cols = []
    for t in tots:
        cols.append((t/max_tot,0,0,t/max_tot))
    
    plt.scatter(x,y,color=cols)
    
    ax = plt.gca()
    ax.set_aspect(1)
    ax.set_ylim(-70,70)
    plt.savefig('out_of_balance.png',dpi=300)
    plt.show()    




plot_mesh(points, edges_b)
plt.show()

points2 = np.transpose(np.rollaxis(np.expand_dims(points, 0),0))

vectors = connections_to_other_points-points2
lengths = np.linalg.norm(vectors,axis=0)
normalised_vectors = vectors/lengths

vectors=np.nan_to_num(vectors, copy=True) 
normalised_vectors=np.nan_to_num(normalised_vectors, copy=True) 
lengths = np.nan_to_num(lengths, copy=True) 

"""Elastic force for non-linear formfinding""" 
def elastic_force(e,EA,L):
    if L!=0:
        f = e*EA/L + 1
    # if f < 0:
    #     f = 0
        return f
    if L == 0:
        return np.nan
    
elastic_force_rule =  np.vectorize(elastic_force)   


v_z = np.array([0,0,1])

e = 0.01
EA = 20
net_fs = elastic_force_rule(e,EA,lengths) #+ v_z
force_vectors = net_fs*normalised_vectors
force_vectors=np.nan_to_num(force_vectors, copy=True) 
net_force_from_bars = np.transpose(np.sum(force_vectors,axis=2))   #we can then use this to find accelerations, velocities, etc


grav_forces = points*0
grav_forces = np.transpose(points*0)
grav_forces[2] = grav_forces[2] + 3
grav_forces = np.transpose(grav_forces)

total_net_force = net_force_from_bars + grav_forces

total_forces = np.linalg.norm(total_net_force,axis=1)
mean_force = np.mean(total_forces)





vs = points*0
ms = points*0+0.4


""" Check if a point is on the boundary """ 
def is_boundary_point(point,mask):
    x = point[0]  + L_width_x/2 #but origin is in centre now!
    y = point[1] + L_width_y/2
    x_ind = int(x*(N_nodes_x-1)/L_width_x)
    y_ind = int(y*(N_nodes_y-1)/L_width_y)
    value = mask[x_ind][y_ind]
    if value == 0:
        return point,np.array([0,0,0])
    if value == 1:
        return np.array([0,0,0]),np.array([1,1,1])

def boundary_points_finder(points,mask):
    global finding_bp
    boundary_points = points*0
    finding_bp = boundary_points*1
    boundary_selector = points*0
    i = 0
    for point in points:
        bp = is_boundary_point(point,mask)
        boundary_points[i] = bp[0]
        boundary_selector[i] = bp[1]
        i+=1
    return boundary_points,boundary_selector


""" Form finding with elastic forces """ 
def run_form_find(points,original_lengths,boundary_points,boundary_selector,mesh_edges):
    vs = points*0
    ms = points*0+0.4
    dt = 0.0001
    
    f_mean = 0
    f_hist = []
    t_hist = []
    #for t in range(0,10000):
    t = 0


    grav_forces = points*0
    grav_forces = np.transpose(points*0)
    grav_forces[2] = grav_forces[2] + 1
    grav_forces = np.transpose(grav_forces)
    global l_g_f
    l_g_f = grav_forces
    while t < 10 or f_mean>0.01:
        
        #p = p*boundary_selector+boundaries
        global points5
        points5 = points
        connections_to_other_points = np.rollaxis(get_point_vectorised(connections_to_other_points_ind),2)+valid_finders

        
        points2 = np.transpose(np.rollaxis(np.expand_dims(points, 0),0))

        vectors = connections_to_other_points-points2
        lengths = np.linalg.norm(vectors,axis=0)
        normalised_vectors = vectors/lengths

        vectors=np.nan_to_num(vectors, copy=True) 
        normalised_vectors=np.nan_to_num(normalised_vectors, copy=True) 
        lengths = np.nan_to_num(lengths, copy=True) 
        e = lengths - original_lengths
        EA = 10
        net_fs = elastic_force_rule(e,EA,lengths) #+ v_z
        global nf
        nf = net_fs
        force_vectors = net_fs*normalised_vectors
        force_vectors=np.nan_to_num(force_vectors, copy=True) 
        net_force_from_bars = np.transpose(np.sum(force_vectors,axis=2))   #we can then use this to find accelerations, velocities, etc

        total_net_force = net_force_from_bars + grav_forces  - 0.5*vs
        global tf
        tf = total_net_force
        total_forces = np.linalg.norm(total_net_force,axis=1)
        global h
        h = f_mean
        f_mean = np.mean(total_forces)
        
        acceleration = total_net_force/ms
        global g_a,g_v
        g_a = acceleration
        g_v = vs
        points = points*1+vs*dt
        vs = vs+acceleration*dt
        points = points*boundary_selector+boundary_points
        
        f_hist.append(f_mean)
        t_hist.append(t)
        if t%100==0:
            plot_views_quick(points)
            plt.show()
        t=t+1
    plot_mesh(points,mesh_edges)
    plot_views(points, mesh_edges, 0)
    plot_views(points, mesh_edges, 1)
    plot_views(points, mesh_edges, 2)
    plot_views(points, mesh_edges, 3)
    plot_views(points, mesh_edges, 4)
    plot_views(points, mesh_edges, 5)


""" Form finding with force density """ 
def run_form_find_force_density(points,original_lengths,these_boundary_points,these_boundary_selector,mesh_edges):
    original_points = points*1
    
    vs = points*0
    ms = points*0+2
    dt = 0.001
    
    f_mean = 0
    f_hist = []
    t_hist = []
    #for t in range(0,10000):
    t = 0
    
    stress_scale = 10000   #stress in kpa
    scaled_bar_stress_array = original_bar_stress_array*1

    grav_forces = points*0
    grav_forces = np.transpose(points*0)
    grav_forces[2] = grav_forces[2] + node_load  #node load in kpa
    grav_forces = np.transpose(grav_forces)
    global l_g_f
    l_g_f = grav_forces
    while t < 10000 or f_mean>node_load/1000:
        
        #p = p*boundary_selector+boundaries
        global points5
        points5 = points
        connections_to_other_points = np.rollaxis(get_point_vectorised(connections_to_other_points_ind),2)+valid_finders
        
        #we need an equivalent for the original stresses
        
        
        
        points2 = np.transpose(np.rollaxis(np.expand_dims(points, 0),0))
        global vectors  #something wrong with vectors? hard to tell
        vectors = connections_to_other_points-points2
        vectors=np.nan_to_num(vectors, copy=True) 
        global z_components
        z_components = vectors[2] #(441,8)  #seems okay...
 
        #new_bars_attempt  #
        #original_stress_state is denoted by bar index...
        #need to get this into array format (441,8)
        net_fs_in_z = scaled_bar_stress_array*z_components/original_lengths   #(441,8)*(441,8)/(441,8) = (441,8)
        global nf  #seems okay
        nf = net_fs_in_z
        
        force_vectors = points2*0
        force_vectors[2] = force_vectors[2] + 1  #so we select the z direction
        force_vectors = force_vectors*net_fs_in_z
        force_vectors=np.nan_to_num(force_vectors, copy=True) 
        global net_force_from_bars   #seems to work now
        net_force_from_bars = np.transpose(np.sum(force_vectors,axis=2))
        
    
        total_net_force = net_force_from_bars + grav_forces  #- 200*vs
        global bar_total_force_for_eq_check
        bar_total_force_for_eq_check = total_net_force
        
        bar_force = np.linalg.norm(net_force_from_bars,axis=1)
        global tf
        
        total_forces = np.linalg.norm(total_net_force,axis=1)*np.transpose(these_boundary_selector)
        tf = total_forces
        global h
        
        f_mean = np.mean(total_forces)   #problem is boundaries have force!
        h = f_mean
        
        total_net_force = total_net_force - 10*vs
        
        acceleration = total_net_force/ms
        global g_a,g_v
        g_a = acceleration
        g_v = vs
        points = points*1+vs*dt
        vs = vs+acceleration*dt
        points = points*these_boundary_selector +these_boundary_points  #why is this not working? I think it is, maybe a plotting problem
        
        f_hist.append(f_mean)
        t_hist.append(t)
        if t%1000==0:
            plot_views_quick(points)
            plt.show()
            print('f_mean',f_mean)
            #wait = input('wait')
        t=t+1
        
        
    plot_mesh(points,mesh_edges)
    plot_views_3D_out_of_balance_force(points,mesh_edges,bar_total_force_for_eq_check,0)
    plot_views(points, mesh_edges, 0)
    plot_views(points, mesh_edges, 1)
    plot_views(points, mesh_edges, 2)
    plot_views(points, mesh_edges, 3)
    plot_views(points, mesh_edges, 4)
    plot_views(points, mesh_edges, 5)


    return points, bar_force


          

    
original_lengths = lengths*1    

""" Find the boundary nodes - those which should be clamped in vertical direction"""
outer_nodes = find_outer_edges_2(points,(L_width_x/2, L_width_x/2),(L_width_x/2, L_width_x/2))
boundary_selector_2 = np.transpose(np.array([outer_nodes]))
boundary_points_2 = (boundary_selector_2)*points
boundary_selector_2 = 1-boundary_selector_2 


#outer_bars
#outer_bars = find_outer_edges_2(mid_points,(L_width_x/2, L_width_x/2),(L_width_x/2, L_width_x/2)) 
#outer_bars_stress = np.transpose(np.array([outer_bars])) - 0.2 #this is the outer bars



left_edge = find_specific_edge(mid_points,'left')
right_edge = find_specific_edge(mid_points,'right')
top_edge = find_specific_edge(mid_points,'top')
bottom_edge = find_specific_edge(mid_points,'bottom')

left_stress = 1
bot_top_stress = (L_width_x/L_width_y)*left_stress
#bot_top_stress = left_stress
#bottom/left = w/H

arch_factor = -0.3
all_factor = 0.10
left_arch_factor = -0.2



""" Make a guess at the desired stress state """ 
outer_bars_stress = left_stress*np.transpose(np.array([left_edge])) + bot_top_stress*np.transpose(np.array([top_edge])) + bot_top_stress*np.transpose(np.array([bottom_edge])) + left_stress*np.transpose(np.array([right_edge])) - all_factor   +arch_factor*np.transpose(np.array([new_arch_bars]))  +left_arch_factor*np.transpose(np.array([left_arch_bars]))  

""" Plot it """ 
plot_frame2D_selfstress(points_2d,edges_b,non_reaction_points,outer_bars_stress,1,True)

""" Project to make it a self stress state """ 
outer_bars_true_self_stress = np.transpose([project_stress(self_stress_states, outer_bars_stress)])

""" Plot it """ 
plot_frame2D_selfstress(points_2d,edges_b,non_reaction_points,outer_bars_true_self_stress,1)

""" scale stress appropriately """ 
outer_bars_true_self_stress = outer_bars_true_self_stress*10000   

""" Create the array required for the form finding """ 
global original_bar_stress_array
original_bar_stress_array = get_original_stress_state(connections_to_other_points_ind, edges_b, outer_bars_true_self_stress, valid_finder) +valid_finders
original_bar_stress_array = original_bar_stress_array*(-1)

""" run the form finding """ 
found_points,bar_force = run_form_find_force_density(points,original_lengths,boundary_points_2,boundary_selector_2,edges_b)

total_bar_force = np.array([bar_force,outer_bars_true_self_stress[0]])

""" Find the total bar forces (from horizontal and vertical components)""" 
def find_actual_bar_forces(points,edges,self_stress_state):
    total_bar_force = []
    i = 0
    for edge in edges:
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        vertical_length = p2[2] - p1[2]
        horizontal_length = ((p2[1] - p1[1])**2+(p2[0] - p1[0])**2)**0.5

        horizontal_force = self_stress_state[i][0]
        vertical_force = horizontal_force*vertical_length/horizontal_length
        
        total_force = np.sign(horizontal_force)*(horizontal_force**2 + vertical_force**2)**0.5
        total_bar_force.append(total_force)
        i+=1
    return total_bar_force

""" Plot some views"""
plot_views_buckled(found_points,edges_b,0)
plot_views_buckled(found_points,edges_b,1)
plot_views_buckled(found_points,edges_b,2)
plot_views_buckled(found_points,edges_b,3)
plot_views_buckled(found_points,edges_b,4)

actual_bar_forces = find_actual_bar_forces(found_points,edges_b,outer_bars_true_self_stress)
plot_views_3D_stress(found_points,edges_b,actual_bar_forces,0)
plot_views_3D_stress(found_points,edges_b,actual_bar_forces,1)
plot_views_3D_stress(found_points,edges_b,actual_bar_forces,2)
plot_views_3D_stress(found_points,edges_b,actual_bar_forces,3)
plot_views_3D_stress(found_points,edges_b,actual_bar_forces,4)


""" Next few sections was an attempt to get forces from an airy stress function"""

""" Equation of plane from three points"""
def plane_from_points(points):
    p1,p2,p3 = points[0], points[1], points[2]
    v1 = np.array(p2) - np.array(p1) 
    v2 = np.array(p3) - np.array(p1) 
    n = np.cross(v1,v2)   #not normalised 
    return n

""" Slope from a vector"""


def slope_from_vector(v):
    v = v/np.linalg.norm(v)
    return v[2]/(v[0]**2 + v[1]**2)**0.5

""" Forces from a given airy stress function (not working)"""    

def forces_from_airy(these_points,edges,cells):
    # not quite fully working yet
    airy_forces = []
    for edge in edges:  #for each edge in the mesh
    
    
        edge_vector = np.array(these_points[edge[1]])-np.array(these_points[edge[0]])

        #need to find the two adjoining cells. Brute force search for now
        
        adjoining_cells = []
        for cell in cells:
            if edge[0] in cell and edge[1] in cell:  #cells gives the three poitns in the cell - so we need to add the points
                adjoining_cells.append(list(cell))
        #we now have the one or two cells
        #print(adjoining_cells)
        if len(adjoining_cells) == 2:   #the edge is in two cells
            if len(adjoining_cells[0]) <= 3 and len(adjoining_cells[1]) <= 3:
                cell_1 = adjoining_cells[0]
                cell_2 = adjoining_cells[1]
                #points_1 = [points[i] for i in cell_1]
                #points_2 = [points[i] for i in cell_2]
                
                p0 = these_points[edge[0]]
                p1 = these_points[edge[1]]
                
                p2 = these_points[list(set(cell_1) - set(edge))[0]]
                p3 = these_points[list(set(cell_2) - set(edge))[0]]
                points_1 = [p0,p1,p2]
                points_2 = [p0,p1,p3]
                
                #print('points 1',points_1)  #points sseem okay...
                #print('points 2',points_2)  #points sseem okay...
                
            if len(adjoining_cells[0]) > 3 or len(adjoining_cells[1]) > 3:  #having some problems with this
                #print('adj',adjoining_cells)
                #print('cell contains 1600')
                cell_1 = adjoining_cells[0]
                cell_2 = adjoining_cells[1]
                if len(cell_1) <=3:
                    hold = cell_2
                    cell_2 = cell_1[:]
                    cell_1 = hold
                #now cell 1 is the arch cell
                p0 = these_points[edge[0]]
                p1 = these_points[edge[1]]
                
                #p2 = (points[20]+points[860])/2   #mid point of tie
                #how about choose somewhere on tie with same x?
                y = (p0[1]+p1[1])/2  #y value of bar mid point
                a = these_points[20]
                b = these_points[860]
                
                p3 = ((y-a[1])/(b[1]-a[1]))*(b-a) + a   #trying  to get a point at similar location along tension tie 
                #print(p2)
                #p2 = these_points[440]
                #p2 = (a+b)/2
                
                
                p2 = these_points[list(set(cell_2) - set(edge))[0]]
                #for points 1 let's use the original 
                points_1 = [p0,p1,p2]
                
                #points_1 = [these_points[20],these_points[440],these_points[860]]  #this worked for stress- but not for finding mid points
                points_2 = [p0,p1,p3]
                
                
            edge_flattened = edge_vector*1
            edge_flattened[2] = 0
            edge_perp = np.cross(edge_flattened,np.array([0,0,1]))
            edge_perp = edge_perp/np.linalg.norm(edge_perp)
            mid_point = (p0+p1)/2
            
            above_point_1 = mid_point + 0.001*edge_perp
            above_point_2 = mid_point - 0.001*edge_perp
            
            
            n1 = plane_from_points(points_1)
            d1 = np.dot(n1,p0)
            
            
            n2 = plane_from_points(points_2)
            d2 = np.dot(n2,p0)
            
            
            #now ensure both vectors point up...
            
            # if n1[2] <0 :
            #     n1 = - n1
            # if n2[2] <0 :
            #     n2 = - n2
            
            proj_point_1_z = (d1-n1[0]*above_point_1[0] - n1[1]*above_point_1[1])/n1[2]
            proj_point_1 = np.array([above_point_1[0],above_point_1[1],proj_point_1_z])
            
            proj_point_2_z = (d2-n2[0]*above_point_2[0] - n2[1]*above_point_2[1])/n2[2]
            proj_point_2 = np.array([above_point_2[0],above_point_2[1],proj_point_2_z])
            
                        
            

            
            #print(change_in_distance)
            v1 = mid_point - proj_point_1
            v2 = proj_point_2 - mid_point
    
            
            s1 = slope_from_vector(v1)
            s2 = slope_from_vector(v2)
            
            change_in_slope = s2-s1    
            #print(s1,s2,change_in_slope)  #these seem too small for found points 2...
            #print(' ')
            
            #change_in_slope = - abs(change_in_slope)  #this seemed to do the trick - but not a viable long term solution...
            airy_forces.append(change_in_slope)
            
        if len(adjoining_cells) == 1:  #bar is on the edge, so only in one cell - the other cell is below
            if len(adjoining_cells[0]) <= 3:
                cell_1 = adjoining_cells[0]
                points_1 = [these_points[i] for i in cell_1]
            
            if len(adjoining_cells[0]) > 3:
                #print('adj',adjoining_cells)
                #for arch normal we want mid arch, and two support points
                #points_1 = [points[20],points[860],points[440]]
                points_1 = [these_points[20],these_points[860],these_points[440]]

                mid_bar = (these_points[20] + these_points[860])/2
                mid_arch = these_points[440]
                slope_vec = mid_arch - mid_bar
                slope = slope_from_vector(slope_vec)
            
            
            
            n1 = plane_from_points(points_1)  #but we need slope, not normal!!
            
            
            slope_vector = np.cross(n1,edge_vector)
            
            
            if slope_vector[2] <0 :
                slope_vector = - slope_vector
                
            s1 = slope_from_vector(slope_vector)

            airy_forces.append(s1)
                
        if len(adjoining_cells) == 0: #the edge is not in a single cell - either none or one of the points is not in a cell...
            print('edge not in cell',edge)
            airy_forces.append(0)
            
    return airy_forces

""" """
def run_form_find_force_density_airy(points,original_lengths,these_boundary_points,these_boundary_selector,mesh_edges,initial_guess):
    original_points = points*1
    
    vs = points*0
    ms = points*0+2
    dt = 0.4
    
    f_mean = 0
    f_hist = []
    t_hist = []
    #for t in range(0,10000):
    t = 0
    
    stress_scale = 10000   #stress in kpa
    scaled_bar_stress_array = original_bar_stress_array*1

    grav_forces = points*0
    grav_forces = np.transpose(points*0)
    node_load = 17.5609756097561
    grav_forces[2] = grav_forces[2] + node_load  #node load in kpa
    grav_forces = np.transpose(grav_forces)
    global l_g_f
    l_g_f = grav_forces
    reset = False
    points = initial_guess
    while t < 10000 or f_mean>node_load/100: #1000 seems reasonable
        
        #p = p*boundary_selector+boundaries
        global points5
        points5 = points
        connections_to_other_points = np.rollaxis(get_point_vectorised(connections_to_other_points_ind),2)+valid_finders
        
        #we need an equivalent for the original stresses
        
        
        
        points2 = np.transpose(np.rollaxis(np.expand_dims(points, 0),0))
        global vectors  #something wrong with vectors? hard to tell
        vectors = connections_to_other_points-points2
        vectors=np.nan_to_num(vectors, copy=True) 
        global z_components
        z_components = vectors[2] #(441,8)  #seems okay...
 
        #new_bars_attempt  #
        #original_stress_state is denoted by bar index...
        #need to get this into array format (441,8)
        net_fs_in_z = scaled_bar_stress_array*z_components/original_lengths   #(441,8)*(441,8)/(441,8) = (441,8)
        global nf  #seems okay
        nf = net_fs_in_z
        
        force_vectors = points2*0
        force_vectors[2] = force_vectors[2] + 1  #so we select the z direction
        force_vectors = force_vectors*net_fs_in_z
        force_vectors=np.nan_to_num(force_vectors, copy=True) 
        global net_force_from_bars   #seems to work now
        net_force_from_bars = np.transpose(np.sum(force_vectors,axis=2))
        
    
        total_net_force = net_force_from_bars + grav_forces  #- 200*vs
        global bar_total_force_for_eq_check
        bar_total_force_for_eq_check = total_net_force
        
        bar_force = np.linalg.norm(net_force_from_bars,axis=1)
        global tf
        
        total_forces = np.linalg.norm(total_net_force,axis=1)*np.transpose(these_boundary_selector)
        tf = total_forces
        global h
        
        f_mean = np.mean(total_forces)   #problem is boundaries have force!
        h = f_mean

        total_net_force = total_net_force - 12*vs
        
        acceleration = total_net_force/ms
        global g_a,g_v
        g_a = acceleration
        g_v = vs
        points = points*1+vs*dt
        vs = vs+acceleration*dt
        
        """
        This is as I was having trouble with stability with these nodes, s
        o I had to use an initial guess where the roof had done a bit of form finding, and then stop these nodes from moving. 
        
        They actually only ended up 1KN out of balance with the vertical loads, so a bodge but not a bad one
        """
        vs[814] = 0
        vs[58] = 0
        
        points = points*these_boundary_selector +these_boundary_points  #why is this not working? I think it is, maybe a plotting problem
        
        f_hist.append(f_mean)
        t_hist.append(t)
        if t % 10==0:
            vs = vs*0
            reset = True
        if t % 1000==0:
            plot_views_quick(points)
            plt.show()
            print('f_mean',f_mean)
            #wait = input('wait')
        t=t+1
        
        
    #plot_mesh(points,mesh_edges)
    #plot_views_3D_out_of_balance_force(points,mesh_edges,bar_total_force_for_eq_check,0)
    plot_views(points, mesh_edges, 0)
    plot_views(points, mesh_edges, 1)
    plot_views(points, mesh_edges, 2)
    plot_views(points, mesh_edges, 3)
    # plot_views(points, mesh_edges, 4)
    # plot_views(points, mesh_edges, 5)


    return points, bar_force   

""" Finding the forces from the original roof as an airy stress function"""
f_airy = forces_from_airy(found_points,edges_b,cells_all)


f_airy_stress = np.transpose(np.array([f_airy]))   #don't scale - the resulting roof will then give us the stresses. But probably need to run it a lot faster

""" Checking if it is a valid self stress state"""
equilibrium_check_airy = is_self_stress_state(eq, f_airy_stress)   #failing.. not a self stress state it seems...
plot_views_quick_out_of_balance(found_points,equilibrium_check_airy)

""" Plotting the forces"""
plot_frame2D_selfstress(points_2d,edges_b,non_reaction_points,f_airy_stress,1,True)

#outer_bars_true_self_stress = outer_bars_stress

""" Projecting onto nullspace and plotting for comparison"""
f_airy_self_stress = np.transpose([project_stress(self_stress_states, f_airy_stress)])
plot_frame2D_selfstress(points_2d,edges_b,non_reaction_points,f_airy_self_stress,1,False)


original_bar_stress_array = get_original_stress_state(connections_to_other_points_ind, edges_b, f_airy_stress, valid_finders) +valid_finders
original_bar_stress_array = original_bar_stress_array*(-1)

""" Using an initial guess to get around stability issues I was having with two nodes"""
initial_guess = np.load('initial_guess_3.npy')
initial_guess = np.transpose(initial_guess)
initial_guess[2] = initial_guess[2]*0.8
initial_guess = np.transpose(initial_guess)

""" Run form finding"""
found_points_2, bar_forces_2 = run_form_find_force_density_airy(points,original_lengths,boundary_points_2,boundary_selector_2,edges_b,initial_guess)


""" Finding the forces from the new airy stress function"""
f_airy = forces_from_airy(found_points_2,edges_b,cells_all)

f_airy_stress = np.transpose(np.array([f_airy]))  #don't scale - the resulting roof will then give us the stresses. But probably need to run it a lot faster

""" Checking if it is a valid self stress state"""
equilibrium_check_airy = is_self_stress_state(eq, f_airy_stress)   #failing.. not a self stress state it seems...
plot_views_quick_out_of_balance(found_points,equilibrium_check_airy)

""" Plotting the forces"""
plot_frame2D_selfstress(points_2d,edges_b,non_reaction_points,f_airy_stress,1,True)

#outer_bars_true_self_stress = outer_bars_stress
""" Projecting onto nullspace and plotting for comparison"""
f_airy_self_stress = np.transpose([project_stress(self_stress_states, f_airy_stress)])
plot_frame2D_selfstress(points_2d,edges_b,non_reaction_points,f_airy_self_stress,1,False)