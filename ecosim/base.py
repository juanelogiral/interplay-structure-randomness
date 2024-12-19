"""
Defines a basic interface for dynamical systems, including
- generate trajectories
- save & load trajectories
"""
import pickle
import warnings
from bisect import bisect_left

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import fftconvolve
from scipy.io import savemat, loadmat


class Storable:
    def save(self, path):
        """Save as a pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path, check_type=False):
        """Load from pickle.

        Can be invoked generically as ``Container.load``, or as
        ``ContainerSubclass.load`` to test explicitly that the pickle is
        of a given subclass.
        """
        with open(path, "rb") as f:
            obs = pickle.load(f)
            if check_type and not isinstance(obs, cls):
                raise RuntimeError(f"data loaded is not a {cls} instance")
            return obs


class DivergenceError(Exception):
    def __init__(self, traj,*args):
        self.traj = traj
        super().__init__(*args)

class Model(Storable):
    """Defines the basic interface for simulation models."""

    def __init__(self, dim, dim_aux=0):
        self.dim = dim
        self.dim_aux = dim_aux

        self.reset_time()

        # main degrees of freedom
        self.vec_x = np.zeros(dim, order='C')
        # time-dependent parameters
        self.vec_y = np.zeros(dim_aux) if dim_aux > 0 else None

        # function used to integrate forward in time
        self._integrate = None

        self.TrajClass = Trajectory

    def reset_time(self):
        self.t = 0
    
    def init_constant(self, c):
        self.vec_x = c*np.ones(self.dim, order='C')

    def run(self, time, record_interval=None, **kwargs):
        """Runs simulation forward by ``time`` time units and returns
        a trajectory object if a ``record_interval`` is specified.

        Notes
        -----
        Returned trajectories include the initial state, but not the final state.
        In this way, joining two consecutive trajectories does not overcount a state
        """
        
        if self._integrate is None:
            raise RuntimeError("No integrator function set")

        vec_t, mat_x, status, message = self._integrate(time, record_interval, **kwargs)

        # I would like to change so that scipy and fixed step integrate are not methods
        #self.t += time
        #self.vec_x = mat_x...
        traj = None
        if record_interval is not None:
            traj = self.TrajClass(vec_t, mat_x, self)
        
        if status == 1:
            raise DivergenceError(traj)
        elif status == -1:
            raise RuntimeError(f"Integrator terminated earlier than expected: {message}")

        if traj is not None:
            return traj
    
    def run_to(self, absolute_time, record_interval=None, **kwargs):
        if absolute_time > self.t:
            ret = self.run(absolute_time - self.t, record_interval, **kwargs)
            self.t = absolute_time #make sure there are no tiny rounding errors
            return ret
        elif absolute_time == self.t:
            return None
        else:
            raise ValueError("Time point is in the past")
       

import numba
import functools
import inspect

class ODEModel(Model):
    """
    Model class that implements an ODE given by the right-hand-side function `f`.

    The *last two* arguments of `f` must be the time and state vector (e.g. named t, x).
    Any prior arguments are interpreted as parameters which will be set as attributes 
    (defaulting to None) in the model object. Keyword arguments matching the parameter 
    names in `f` can be given at initialization. 
    
    The function `f` will be just-in-time compiled with numba for efficiency.
    
    `J`, if supplied, is a function with the same argument structure as `f` which should
    give the Jacobian matrix of `f`

    If `trans` is given (a tuple of a funciton and its inverse), then the the ODE state 
    will first be transformed before it is evolved (by the corresponding transformed `f_trans`)
    and then de-transformed.
    
    """

   
    def __init__(self, S, f, J=None, trans=None, f_trans = None, J_trans= None, njit=True, **kwargs):
        super().__init__(S)
        
        # Check compatibility of arguments
        f_param_names = inspect.getfullargspec(f)[0][:-2]
        param_defaults = {k : None for k in f_param_names}
        if set(kwargs.keys()).issubset(f_param_names):
            param_defaults.update(kwargs)
        else:
            wrong_args = set(kwargs.keys()).difference(f_param_names)
            raise ValueError(f"Argument(s) {wrong_args} not parameters of f")
        
        # Set attributes
        self._ode_param_names = param_defaults.keys() #remember which attributes are model parameters
        for key, val in param_defaults.items():
            if not hasattr(self,key):
                setattr(self,key,val)
            else:
                raise ValueError("Attempted overwrite of pre-defined attribute")

        self._f = numba.njit(f) if njit else f
        self._J = numba.njit(J) if (J is not None and njit) else J

        # Set transformed functions
        self._trans = trans
        if trans is not None:
            self._trans = trans
            self._f_trans = numba.njit(f_trans) if njit else f_trans
            self._J_trans = numba.njit(J_trans) if (J_trans is not None and njit) else J_trans

        self._refresh_integrator()

        self.integration_options = {} #Default values
        
        
    def _scipy_integrate(self, f, time, /, record_interval=None, **kwargs):
        """Runs the simulation forward using scipy ode solver

        ``trans`` is a pair of functions, transformation and  anti-transformation, applied
        to the state before and after simulation
        """

        trans = self._trans if self._trans is not None else (lambda x: x, lambda x: x)
        
        if record_interval is not None:
            vec_t = np.arange(self.t, self.t + time, record_interval)
        else:
            vec_t = np.array([self.t, self.t + time])
        
        solverkwargs = {}
        solverkwargs.update(self.integration_options)
        solverkwargs.update(kwargs)
        solverkwargs.update({"t_eval": vec_t})
        
        vec_x = trans[0](self.vec_x)
        
        sol = solve_ivp(f, [self.t, self.t + time], vec_x, **solverkwargs)

        self.t += time
        self.vec_x = np.ascontiguousarray(trans[1](sol.y[:, -1]))  # update state of simulation!
        
        vec_t = sol.t
        mat_x = trans[1](sol.y.T)
        return vec_t, mat_x, sol.status, sol.message

    def set_integration_options(self,**kwargs):
        """
        These keyword arguments will be fed to the ode integrator unless overriden
        """    
        self.integration_options.update(kwargs)

    def _refresh_integrator(self):
        """Necessary so that parameters as attributes can be baked into integrator with numba"""
        ode_param_values = [getattr(self,key) for key in self._ode_param_names]
        if self._trans is not None:
            f = functools.partial(self._f_trans, *ode_param_values)
        else:
            f = functools.partial(self._f, *ode_param_values)

        self._integrate = functools.partial(self._scipy_integrate, f)

    def run(self, time, record_interval=None,**kwargs):
        self._refresh_integrator()
        return super().run(time, record_interval=record_interval,**kwargs)

    def fun(self, vec_x=None):
        """
        Evaluates f at the current state vector and 
        parameters
        """
        if vec_x is None : vec_x = self.vec_x
        ode_param_values = [getattr(self,key) for key in self._ode_param_names]
        return self._f(*ode_param_values, self.t, vec_x)

    def jacobian(self, vec_x=None):
        """
        Return the jacobian matrix evaluated at the current state vector and 
        parameters
        """
        if self._J is None:
            return None
        if vec_x is None : vec_x = self.vec_x
        params = [getattr(self,k) for k in self._ode_param_names]
        return self._J(*params, self.t, vec_x)



class DiscreteTimeModel(Model):
    """
    TODO Finish implementation
    """

    def _fixed_step_integrate(self, f_step, time, /, record_interval=None):
        """Runs similation forward by (continuous) time ``time`` using ``f_step``
        step function advancing in units of ``self.dt``.

        Notes
        -----
        Requires ``self.dt`` to be set prior.

        If ``time / record_interval`` is not an integer number
        of ``dt``s integration will not fully integrate a duration of ``time``.
        """
        if time < self.dt:
            raise ValueError("Integration time shorter than one step")

        steps = int(np.rint(time / self.dt))

        if record_interval is None:
            n_intrvls = 1
            rec_intrvl_steps = steps
        else:
            rec_intrvl_steps = int(np.rint(record_interval / self.dt))
            n_intrvls = steps // rec_intrvl_steps
            # steps_diff = steps - n_intrvls * rec_intrvl_steps

        t0 = self.t
        vec_t = t0 + self.dt * rec_intrvl_steps * np.arange(n_intrvls)
        mat_x = np.zeros((n_intrvls, self.dim))
        status = 0
        message = "Integrator terminated correctly."
        try:
            for i in range(n_intrvls):
                mat_x[i, :] = self.vec_x
                for s in range(rec_intrvl_steps):
                    self.vec_x = f_step(self.vec_x, self.t)
                    self.t = (
                        t0 + (i * n_intrvls + s) * self.dt
                    )  # less rounding error than adding increments
        except BaseException as e:
            status = 1
            message = e.args
            vec_t = vec_t[: mat_x.shape[0]]
        return vec_t, mat_x, status, message



class Trajectory(Storable):
    """Stores the outcome

    Attributes
    ----------
    vec_t:
        array of time points
    mat_x:
        time series with time along row axis (axis=0) and state vectors along
        columns (axis=1)
    S:
        dimension of state vector
    N:
        number of time points
    T:
        total duration

    """
    
    """
        Because this class admits slicing, we reproduce the behavior of Numpy arrays under slicing.
        When calling traj[1:10], for example, the returned object is a view onto the original trajectory, not a copy.
        This is to reduce memory usage and time complexity when only wants to reduce the range of a Trajectory, e.g. for
        plotting.
        To do a copy, use Trajectory.copy().
        
        The attribute 'base' is either 'no_base' or a Trajectory object. If it is 'no_base' then the Trajectory is not a view,
        otherwise it points to the Trajectory from which it is a view.
    """
    
    @classmethod
    def create_view(cls,traj,slice_to_be_kept):
        '''Create a view of a given trajectory'''
        new_traj = Trajectory(slice_to_be_kept,as_view_of=traj)
        return new_traj
        
    def copy(self):
        '''Create a copy of a given trajectory'''
        new_vec_t, new_mat_x = self._return_time_points_inside_domain(self.time_domain[0], self.time_domain[1])
        new_traj = Trajectory(new_vec_t,new_mat_x)
        new_traj._sim_class = self.sim_class
        return new_traj
    
    def __init__(self, *args,**kwargs):
        '''Constructor
        If as_view_of is an argument then call should be (slice_to_be_kept,**,as_view_of,parent_sim=None)
        Otherwise vec_t, mat_x, parent_sim=None
        '''
        if 'as_view_of' not in kwargs:
            vec_t =args[0]
            mat_x=args[1]
            if vec_t.shape[0] != mat_x.shape[0]:
                raise ValueError(f"Inconsistent time and state arrays: {vec_t.shape[0]} and {mat_x.shape[0]}")
            self._vec_t = vec_t
            self._mat_x = mat_x
            self.base = 'no_base'
            
        elif isinstance(kwargs['as_view_of'],Trajectory):
            
            self.base = kwargs['as_view_of']
            if len(args)>2:
                raise IndexError(f"Trajectories support 1-d (only time) or 2-d (time and species) indexing, but {len(args)} were provided.")
            if len(args)==2:
                time_idx = args[0]
                sp_idx = args[1]
                if isinstance(sp_idx,int):
                    if sp_idx < 0:
                        sp_idx += self.base.S
                    sp_idx = np.array([sp_idx])
                elif isinstance(sp_idx,slice):
                    sp_idx_start = sp_idx.start
                    sp_idx_stop = sp_idx.stop
                    if sp_idx.start == None:
                        sp_idx_start = 0
                    elif sp_idx.start < 0:
                        sp_idx_start += self.base.S
                    if sp_idx.stop == None:
                        sp_idx_stop = self.base.S
                    elif sp_idx.stop < 0:
                        sp_idx_stop += self.base.S
                
                    sp_idx = np.array(range(sp_idx_start,sp_idx_stop))
                elif '__iter__' in dir(sp_idx):
                    sp_idx = np.array(list(sp_idx))
                else:
                    raise IndexError(f"Species index should be int, iterable or slice, not {type(sp_idx)}.")
            elif len(args)==1: #only time slicing
                time_idx = args[0]
                sp_idx = np.array(range(self.base.S))
            else: #no time and no species slicing, the view is the same as the original
                sp_idx = np.array(range(self.base.S))
                time_idx = slice(None,None)
            
            time_idx_start = time_idx.start
            time_idx_stop = time_idx.stop
            if time_idx.start == None:
                time_idx_start = self.base.time_domain[0]
            
            if time_idx.stop == None:
                time_idx_stop = self.base.time_domain[1]
                
            self._time_domain = (time_idx_start,time_idx_stop)
            
            self._min_idx,self._max_idx = self.base._find_time_domain_indices(time_idx_start,time_idx_stop)
            
            self.sp_domain = np.array(range(0,self.base.S))[sp_idx]
            '''BEWARE: Numpy does not check for bounds when slicing, so no error will be thrown if sp_idx contains
            species beyond the number of species in self.
            '''
            if not len(self.sp_domain):
                raise ValueError("Cannot slice an empty set of species.")
        else:
            raise ValueError(f"as_view_of must be of type Trajectory, not {type(kwargs['as_view_of'])}")
        self._sim_class = type(kwargs['parent_sim']) if 'parent_sim' in kwargs else None
    
    def __setattr__(self,name,val):
        if name == 'base' and hasattr(self,'base') and not self.base is None: #We mimick the behavior of numpy arrays
            raise AttributeError("Attribute 'base' of Trajectory is not writable.")
        super().__setattr__(name,val)
    
    def __repr__(self):
        return f"Trajectory (type {self.sim_class}) - from t = {self.time_domain[0]} to t = {self.time_domain[-1]} - {self.S} dof"
    
    def _find_time_domain_indices(self,min_time,max_time=None):
        def take_closest(myList, myNumber):
            ## https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            """
            Assumes myList is sorted. Returns closest index to myNumber from below
            """
            if myNumber < myList[0] or myNumber > myList[-1]:
                raise ValueError("Time is outside the range of the Trajectory.")
            pos = bisect_left(myList, myNumber)
            if pos == 0: # this can only happen if myNumber == myList[0]
                return 0
            return pos-1
        
        vec_t = self.vec_t
        idx1 = take_closest(vec_t, min_time)
        if max_time is not None:
            idx2 = take_closest(vec_t, max_time)
        
        return idx1,idx2 if max_time is not None else idx1
    
    def _return_time_points_inside_domain(self,min_time,max_time,bound_idx = None):
        ''' Subsamples vec_t and mat_x in the given time_domain.
            Used by __get_item__ subfunctions and __add__
            bound_idx allows to bypass the calculation of the nearest time point indices if these are already known (this is to allow for caching)
        '''
        
        if self.base !='no_base':
            
            if min_time > max_time:
                raise ValueError(f"({min_time},{max_time}) is not a valid time range.")
            elif min_time < self.time_domain[0] or max_time > self.time_domain[1]:
                raise ValueError("Time is outside the range of the trajectory.")
            vect,matx = self.base._return_time_points_inside_domain(min_time,max_time,None if bound_idx==None else (bound_idx[0]+self._min_idx,bound_idx[1]+self._min_idx))
            matx = matx[:,self.sp_domain]
            return vect,matx
        
        if bound_idx is None:
            idx1,idx2 = self._find_time_domain_indices(min_time,max_time)
        else:
            idx1,idx2 = bound_idx
        #we interpolate the values of the requested endpoints so that the trajectory really starts and ends there
        #and not at the closest points
        new_t = self.vec_t[idx1+1:idx2+1]
        new_m = self.mat_x[idx1+1:idx2+1]
        if self.vec_t[idx1+1] != min_time:
            new_t = np.concatenate([[min_time],new_t])
            mat_x_1 = self.mat_x[idx1] + (self.mat_x[idx1+1]-self.mat_x[idx1]) * (min_time - self.vec_t[idx1])/(self.vec_t[idx1+1] - self.vec_t[idx1])
            new_m = np.concatenate([[mat_x_1],new_m])
            
        new_t = np.concatenate([new_t,[max_time]])
        mat_x_2 = self.mat_x[idx2] + (self.mat_x[idx2+1]-self.mat_x[idx2]) * (max_time - self.vec_t[idx2])/(self.vec_t[idx2+1] - self.vec_t[idx2])
        new_m = np.concatenate([new_m,[mat_x_2]])
        
        return new_t,new_m
    
    def __add__(self,obj):
        if not isinstance(obj, Trajectory):
            raise ValueError(f"Cannot add Trajectory to object of type {type(obj)}.")
        if self.sim_class != obj.sim_class:
            warnings.warn(
                "Adding trajectories with different types of parent simulations.",
                RuntimeWarning(),
            )
        if self.S != obj.S:
             raise ValueError(f"Cannot add Trajectory with {self.S} dof to Trajectory with {obj.S} dof.")
        if self.time_domain[1] >= obj.time_domain[0]:
            raise ValueError(f"Cannot add trajectories if time intervals are not monotonically arranged, ({self.time_domain[1]} >= {obj.time_domain[0]}).")
        
        vec_t_1,mat_x_1 = self._return_time_points_inside_domain(self.time_domain[0], self.time_domain[1])
        vec_t_2,mat_x_2 = obj._return_time_points_inside_domain(obj.time_domain[0], obj.time_domain[1])
        
        vec_t = np.concatenate([vec_t_1,vec_t_2])
        mat_x = np.concatenate([mat_x_1,mat_x_2])
        tr = Trajectory(vec_t,mat_x)
        tr._sim_class = self.sim_class
        return tr 
    
    def _get_as_item(self,time_idx,sp_idx):
        ''' This function is called by __getitem__
            THIS RETURNS A SINGLE-TIME VALUE
        '''
        
        def take_closest(myList, myNumber):
            ## https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            """
            Assumes myList is sorted. Returns closest index to myNumber from below
            """
            if myNumber < myList[0] or myNumber > myList[-1]:
                raise ValueError("Time is outside the range of the Trajectory.")
            pos = bisect_left(myList, myNumber)
            if pos == 0:  # this can only happen if myNumber == myList[0]
                return 0
            return pos-1
        
        if self.base != 'no_base' and (time_idx < self.time_domain[0] or time_idx > self.time_domain[1]):
            raise ValueError("Time is outside the range of the Trajectory.")
        
        if self.base != 'no_base':
            real_sp_idx = self.sp_domain[sp_idx]
        else:
            real_sp_idx = sp_idx
        
        idx = take_closest(self.vec_t, time_idx)
        return self.mat_x[idx,real_sp_idx] + (self.mat_x[idx+1,real_sp_idx]-self.mat_x[idx,real_sp_idx]) * (time_idx - self.vec_t[idx])/(self.vec_t[idx+1] - self.vec_t[idx])
        
    def _get_as_copy(self,time_idx,sp_idx):
        ''' This function is called by __getitem__
            THIS RETURNS A COPY OF THE TRAJECTORY
        '''
        if time_idx.start == None:
            time_idx_start = self.time_domain[0]
        else:
            time_idx_start = time_idx.start
        if time_idx.stop == None:
            time_idx_stop = self.time_domain[1]
        else:
            time_idx_stop = time_idx.stop
        
        if time_idx_start >= time_idx_stop:
            raise ValueError(f"({time_idx_start},{time_idx_stop}) is not a valid time range.")
        
        if self.base != 'no_base' and (time_idx_start < self.time_domain[0] or time_idx_stop > self.time_domain[1]):
            raise ValueError("Time is outside the range of the Trajectory.")
        
        if self.base != 'no_base':
            real_sp_idx = self.sp_domain[sp_idx]
        else:
            real_sp_idx = sp_idx
        
        new_t,new_m = self._return_time_points_inside_domain(time_idx_start, time_idx_stop)
        tr = Trajectory(new_t,new_m[:,real_sp_idx])
        tr._sim_class = self.sim_class
        return tr
        
    def _get_as_view(self,time_idx,sp_idx):
        ''' This function is called by __getitem__
            
            THIS RETURNS A VIEW OF THE TRAJECTORY
        '''
        if time_idx.start == None:
            time_idx_start = self.time_domain[0]
        else:
            time_idx_start = time_idx.start
        if time_idx.stop == None:
            time_idx_stop = self.time_domain[1]
        else:
            time_idx_stop = time_idx.stop
        
        if time_idx_start >= time_idx_stop:
            raise ValueError(f"({time_idx_start},{time_idx_stop}) is not a valid time range.")
        
        if self.base != 'no_base' and (time_idx_start < self.time_domain[0] or time_idx_stop > self.time_domain[1]):
            raise ValueError("Time is outside the range of the Trajectory.")
        tr = Trajectory(time_idx,sp_idx,as_view_of=self)
        tr._sim_class = self.sim_class
        return tr
    
    def __getitem__(self,val):
        ''' If val is a tuple, the first one represents time and the second species. For time: 
                if val is float it interpolates between mat_x[i] and mat_x[i+1] where vec_t[i] is closest to val below it.
                if val is a pair it returns mat_x[i:j] where vec_t[i], vec_t[j] are defined as previously
                if val = 'last' it gives the last recorded time
            Otherwise only time is considered
            
            THIS DISTRIBUTES THE CALL DEPENDING ON THE TYPE OF TIME SLICING
        '''
        if isinstance(val,tuple):
            if len(val)>2:
                raise IndexError(f"Trajectories support 1-d (only time) or 2-d (time and species) indexing, but {len(val)} were provided.")
            time_idx = val[0]
            sp_idx = val[1]
            if isinstance(sp_idx,int):
                if sp_idx < 0:
                    sp_idx += self.base.S
                    sp_idx = np.array([sp_idx])
            elif isinstance(sp_idx,slice):
                sp_idx_start = sp_idx.start
                sp_idx_stop = sp_idx.stop
                if sp_idx.start == None:
                    sp_idx_start = 0
                elif sp_idx.start < 0:
                    sp_idx_start += self.S
                if sp_idx.stop == None:
                    sp_idx_stop = self.S
                elif sp_idx.stop < 0:
                    sp_idx_stop += self.S
            
                sp_idx = np.array(range(sp_idx_start,sp_idx_stop))
            elif '__iter__' in dir(sp_idx):
                sp_idx = np.array(list(sp_idx))
            else:
                raise IndexError(f"Species index should be int, iterable or slice, not {type(sp_idx)}.")
        else:
            time_idx = val
            sp_idx = np.array(range(self.S))
        
        if isinstance(time_idx, slice):
            return self._get_as_view(time_idx,sp_idx)
        elif isinstance(time_idx, (int, float)) and not isinstance(time_idx, bool):
            return self._get_as_item(time_idx, sp_idx)
        elif time_idx == 'last':
            
            if self.base != 'no_base':
                real_sp_idx = self.sp_domain[sp_idx]
                return self.base[self.time_domain[-1],real_sp_idx]
            else:
                real_sp_idx = sp_idx
                return self.mat_x[-1,real_sp_idx]
        else:
            raise ValueError(f"Trajectory indices must be 'last', numeric or slice, not {type(val)}.")

    def add_params(self, params={}):
        for key, val in params.items():
            setattr(self, key, val)

    
    def __floordiv__(self,fun):
        return Trajectory.by_transformation(self,fun)
    
    @staticmethod
    def by_transformation(traj, f, *args, **kwargs):
        anchors = traj.anchors
        return Trajectory(anchors[0], f(anchors[1],*args, **kwargs))
    
    '''
    apply and map modify a trajectory by applying a function to it:
     -> apply allows for arbitrary functions that mix different times
     -> map acts time-wise, so it does transformations that are 'diagonal' in time.
    '''        
    def apply(self,fun,*args,**kwargs):
        # Transforms the trajectory in place by applying `fun` 
        # return Trajectory(self.vec_t, fun(self.mat_x,*args,**kwargs))
        self.mat_x = fun(self.mat_x, *args, **kwargs)

    def map(self,fun,*args,**kwargs):
        # Returns a trajectory that is a transformation of the original one by the function fun
        anchors = self.anchors
        return Trajectory(anchors[0],np.array(list(map(lambda x: fun(x,*args,**kwargs),anchors[1]))))
    
    def smooth(self,smooth_d=5,window='flat'):
        # Returns a trajectory that is a smoothed version of the original one
        if smooth_d==0:
            return self
        anchors = self.anchors
        vec_t = anchors[0]
        mat_x = anchors[1]
        if window == 'flat': #moving average
            w = np.ones(smooth_d*2+1,'d')/(smooth_d*2+1)
        else:
            raise ValueError("Unknown window type.")
        
        new_vec_t = vec_t[smooth_d:-smooth_d]
        new_mat_x = np.array([fftconvolve(_, w, mode='valid') for _ in mat_x.T]).T
        return Trajectory(new_vec_t,new_mat_x)
    
    def extract(self, iterable):
        #Returns a trajectory with lesser anchors than the initial trajectory
        return Trajectory(np.array(iterable),np.array([self[t] for t in iterable]))
    
    @property
    def real(self):
        anchors = self.anchors
        new_vec_t = anchors[0]
        new_mat_x = anchors[1].real
        return Trajectory(new_vec_t,new_mat_x)
    
    @property
    def imag(self):
        anchors = self.anchors
        new_vec_t = anchors[0]
        new_mat_x = anchors[1].imag
        return Trajectory(new_vec_t,new_mat_x)
    
    def angle(self):
        anchors = self.anchors
        new_vec_t = anchors[0]
        new_mat_x = anchors[1].angle
        return Trajectory(new_vec_t,new_mat_x)
    
    @property
    def differential(self):
        # Returns a trajectory that is the derivative of the original one.
        anchors = self.anchors
        new_vec_t = anchors[0]
        new_mat_x = np.diff(anchors[1],axis=0)/np.diff(anchors[0])[:,None]
        new_mat_x = np.concatenate([new_mat_x,[new_mat_x[-1]]])  #To keep the same time domain as initially we double the last value
        return Trajectory(new_vec_t,new_mat_x)

    def time_correlations(self):
        mat_x = self.mat_x
        return np.corrcoef(mat_x.T)

    @property
    def vec_t(self):
        if self.base == 'no_base':
            return self._vec_t
        else:
            return self.anchors[0]
            
    @property
    def mat_x(self):
        if self.base == 'no_base':
            return self._mat_x
        else:
            return self.anchors[1]
            
    @property
    def sim_class(self):
        if self.base == 'no_base':
            return self._sim_class
        else:
            return self.base.sim_class
    
    @property
    def time_domain(self):
        if self.base == 'no_base':
            return (self.vec_t[0],self.vec_t[-1])
        else:
            return self._time_domain
    
    @property
    def anchors(self):
        if self.base != 'no_base':
            return self._return_time_points_inside_domain(self.time_domain[0], self.time_domain[1],bound_idx = (0,self._max_idx-self._min_idx))
        else:
            return self.vec_t,self.mat_x
    
    @property
    def S(self):
        if self.base == 'no_base':
            return self.mat_x.shape[1]
        else:
            return len(self.sp_domain)

    @property
    def N(self):
        return len(self.anchors)

    @property
    def T(self):
        return self.time_domain[1]-self.time_domain[0]
    
    @classmethod
    def loadmat(self,path):
        """
        Load from .mat format
        """
        with open(path, "rb") as f:
            dat = loadmat(f)
            return Trajectory(dat['vec_t'][0], dat['mat_x'])

    def variation(self):
        vec_t,mat_x = self.anchors
        return np.trapz(mat_x**2,vec_t,axis=0) - np.trapz(mat_x,vec_t,axis=0)**2 / (vec_t[-1]-vec_t[0])

    def savemat(self,path,meta=None,vars={}):
         """
         Save to .mat format
         """
         with open(path, "wb") as f:
            fields = {'vec_t' : self.vec_t, 'mat_x' : self.mat_x}
            if meta is not None:
                fields.update({'meta' : meta})
            fields.update(vars)
            savemat(f, fields)

    
    

# def mat_part(dat):
#     """
#     Returns `dat.mat_x` if `dat` is `Trajectory`, otherwise returns `dat` 
#     """
#     if type(dat) == Trajectory:
#         return dat.mat_x
#     elif type(dat) == np.ndarray:
#         return dat
#     else:
#         raise ValueError("Unexpected argument type")