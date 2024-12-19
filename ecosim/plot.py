"""

"""

import colorsys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.random import default_rng
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

def randgroupcolor(n_group,n_per_group,seed=None,type='bright',cmap=False,**kwargs):
    """Returns ``n_group`` * ``n_per_group`` random colours such that colors in a group are coherent
    type: {bright, soft}
    """
    color_rng = default_rng(seed)

    if type not in ("bright","soft"):
        raise ValueError("Please choose 'bright' or 'soft' for type")

    if type =='bright':
        low,high = .9,1
    elif type=='soft':
        low,high=.5,.8
    # Generate color map for bright colors, based on hsv
    if 'color_offset' not in kwargs:
        offset = np.random.uniform(0,1)
    else:
        offset = kwargs['color_offset']
    grp_HV = [
        (
            (i/n_group + offset),
            color_rng.uniform(low=low, high=high),
        )
        for i in range(n_group)
    ]
    eps = .2/n_group    
    colors_HSV = [((h+np.random.uniform(-eps,eps))%1,s,v) for h,v in grp_HV for s in np.concatenate([[.9],np.random.uniform(.2,.9,n_per_group-1)])]

    colors_RGB = []
    for c in colors_HSV:
        colors_RGB.append(colorsys.hsv_to_rgb(c[0], c[1], c[2]))

    if cmap:
        return LinearSegmentedColormap.from_list(f"randcolor_{type}", colors_RGB, N=N)
    else:
        return colors_RGB

def randcolor(N, seed=None, type="bright", cmap=False):
    """Returns ``N`` random colours
    type: {bright, soft}
    """

    color_rng = default_rng(seed)

    if type not in ("bright", "soft"):
        raise ValueError("Please choose 'bright' or 'soft' for type")

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        colors_HSV = [
            (
                color_rng.uniform(low=0.0, high=1),
                color_rng.uniform(low=0.2, high=1),
                color_rng.uniform(low=0.9, high=1),
            )
            for i in range(N)
        ]

        colors_RGB = []
        for c in colors_HSV:
            colors_RGB.append(colorsys.hsv_to_rgb(c[0], c[1], c[2]))

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.15
        high = 0.95
        colors_RGB = [
            (
                color_rng.uniform(low=low, high=high),
                color_rng.uniform(low=low, high=high),
                color_rng.uniform(low=low, high=high),
            )
            for i in range(N)
        ]
    if cmap:
        return LinearSegmentedColormap.from_list(f"randcolor_{type}", colors_RGB, N=N)
    else:
        return colors_RGB


def lineplot(traj, log=False, ax=None, color_seed=None,factor_in_groups=None,color_type='bright',**kwargs):
    """ """
    if ax is None:
        fig, ax = plt.subplots()

    mat_x = traj.mat_x
    vec_t = traj.vec_t

    if factor_in_groups is None:
        colors = randcolor(mat_x.shape[1], seed=color_seed,type=color_type)
    else:
        colors = randgroupcolor(factor_in_groups,mat_x.shape[1]//factor_in_groups, seed=color_seed,type=color_type,**kwargs)
    for i in np.random.permutation(range(traj.S)):
        ax.plot(vec_t, mat_x[:, i], color=colors[i])

    if log:
        ax.set_yscale("log")
        
    return ax

def lineplot2d(traj1,traj2, log=False, ax=None, color_seed=None,factor_in_groups=None,color_type='bright',**kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    mat_x1 = traj1.mat_x
    mat_x2 = traj2.mat_x

    if mat_x1.shape != mat_x2.shape:
        raise ValueError("Trajectories should have the same number of dof and the same number of time points.")

    if factor_in_groups is None:
        colors = randcolor(mat_x1.shape[1], seed=color_seed,type=color_type)
    else:
        colors = randgroupcolor(factor_in_groups,mat_x1.shape[1]//factor_in_groups, seed=color_seed,type=color_type,**kwargs)
    for i in range(traj1.S):
        ax.plot(mat_x1[:,i], mat_x2[:, i], color=colors[i])

    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")
        
    return ax

def stackplot(traj, ax=None, color_seed=None,factor_in_groups=None,color_type = 'bright',**kwargs):
    """ """
    if ax is None:
        fig, ax = plt.subplots()
    
    mat_x = traj.mat_x
    vec_t = traj.vec_t
    
    if factor_in_groups is None:
        colors = randcolor(mat_x.shape[1], seed=color_seed,type=color_type)
    else:
        colors = randgroupcolor(factor_in_groups,mat_x.shape[1]//factor_in_groups, seed=color_seed,type=color_type,**kwargs)
    
    ax.stackplot(vec_t, mat_x.T, colors=colors)

    return ax


def phi_abundanceplot(traj, ax=None):
    """
    Plots each species as a dot in the (abundance,phi) plane. This is relevant mainly in the equilibrium phase.
    Parameters
    ----------
    traj, ax
    traj should contain the list of phi's for each species as an attribute.

    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(traj.getattr("phi"), traj.mat_x[-1])

    return ax




def mesh_from_columns(vec_x, vec_y, vec_z, mask_nan=False):
    """
    Takes matching vectors for x, y, z=f(x,y) data, and returns mesh grid matrices X,Y,Z 
    """
    m = {(vec_x[i],vec_y[i]) : vec_z[i] for i in range(len(vec_x))}
    vec_x_ord = np.unique(vec_x)
    vec_y_ord = np.unique(vec_y)
    X,Y = np.meshgrid(vec_x_ord,vec_y_ord)
    
    Z = np.full(X.shape, np.NaN)

    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            Z[j,i] = m[(X[0,i]),(Y[j,0])]
    
    if mask_nan:
        Z = ma.masked_where(np.isnan(Z), Z)
        Y = ma.masked_where(np.isnan(Y), Y)
        X = ma.masked_where(np.isnan(X), X)

    return X,Y,Z

def colorbar(mappable, extend='neither'):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return fig.colorbar(mappable, cax=cax, extend=extend)


def meshplot(vec_x, vec_y, vec_z, ax=None, cbar=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    X,Y,Z = mesh_from_columns(vec_x, vec_y, vec_z)
    pc = ax.pcolormesh(X,Y,Z,**kwargs)
    
    if cbar:
        vmin = kwargs['vmin'] if ('vmin' in kwargs) else None
        vmax = kwargs['vmax'] if ('vmax' in kwargs) else None
        
        if vmax is None:
            if vmin is None:
                colorbar(pc, extend = 'neither')
            else:
                colorbar(pc, extend = 'min')
        else:
            if vmin is None:
                colorbar(pc, extend = 'max')
            else:
                colorbar(pc, extend = 'both')

    return ax



if __name__ == '__main__':
    """
    Command line utility that takes a plotting command, path to a trajectory, and optionally an
    output location, and generates a plot of the trajectory with the default settings. By default
    it is saved in the pwd from which the call is made with <plottype>_ tacked onto the file name 
    of the trajectory. 
    """
    import argparse
    import inspect
    import os
    import base
    currentdir = os.path.dirname(os.path.realpath(__file__))
    from os.path  import join as joinpath
    def subpath(*args): return joinpath(currentdir, *args)

    parser = argparse.ArgumentParser()
    parser.add_argument("plot_type",choices=['stackplot','lineplot'])
    parser.add_argument("trajectory")
    parser.add_argument("-o", default=None)
    args = parser.parse_args()
      
    def get_function_by_name(s):
        global_symbols = globals()
        if s in global_symbols and inspect.isfunction(global_symbols[s]):
            return global_symbols[s]
    
    fun = get_function_by_name(args.plot_type)

    traj = base.Trajectory.load(args.trajectory)
    
    ax = fun(traj)

    if args.o is None:
        trajname_with_extension = os.path.basename(args.trajectory)
        trajname, _ = os.path.splitext(trajname_with_extension)
        cwd = os.getcwd()
        path = joinpath(args.plot_type + "_" + trajname + ".png")
    else:
        path = args.o

    ax.figure.savefig(path)    
    
    
