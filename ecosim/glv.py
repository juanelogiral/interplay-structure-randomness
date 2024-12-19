"""
Generalized Lotka-Volterra model


Testing that must be done
Do we need to rerun set-integrator if vec_K etc change? if they are integer (so not ref)?
"""

import functools
import inspect

import numba
import numpy as np

from . import base


class LotkaVolterra(base.ODEModel):
    """
    Generalized Lotka-Volterra with species-specific but time-constant parameters:
    ``dx_i/dt = lam_i/K_i + r_i * x_i * (1 - sum_j (alpha)_ij * K_j/K_i * x_j)``

    Notes
    -----
    Internally, the dynamics in represented as
    ``dx_i/dt = c_i + k_i * x_i +  sum_j x_i b_ij x_j``

    """

    def __init__(self, S, trans = 'default'):
        
        if trans == 'default':
            trans = 'log'
        self._trans_string = trans
            
        def f(c,k,mat_b,t,x):            
            return c + x * (k + (mat_b@x)) 
        
        def J(c,k,mat_b,t,x):
            x_T = x.reshape(-1, 1)           
            jac = mat_b * x_T
            diag = np.diag(mat_b)*x + k + mat_b @ x
            np.fill_diagonal(jac, diag)
            return jac

        if trans is None:
            super(LotkaVolterra, self).__init__(S, f, J)
        elif trans == 'log':
            
            trans_pair = (np.log, np.exp)

            def f_trans(c,k,mat_b,t,y):
                if np.amax(c) == 0:
                        return k + (mat_b@ np.exp(y))
                else:
                    return c*np.exp(-y) + k + mat_b@ np.exp(y)
                    
            def J_trans(c,k,mat_b,t,y):
                y_T = y.reshape(-1, 1)           
                jac = mat_b * np.exp(y_T)
                diag = np.diag(mat_b)*np.exp(y) + k + mat_b @ np.exp(y)
                np.fill_diagonal(jac, diag)
                return jac * np.exp(-y)

            super(LotkaVolterra, self).__init__(S, f, J, trans=trans_pair, f_trans=f_trans, J_trans=J_trans)
        else:
            raise ValueError(f"Unknown transformation rule: {trans}")

        # Main parameterization
        self.mat_b = np.zeros((S, S))  # x_i x_j coeff
        self.k = 1  # x_i coeff
        self.c = np.array(0)  # constant coeff; array needed for numba compatibility

        # UI parametrization
        self._lam = 0
        self._K = 1
        self._r = 1
        # no need to store mat_alpha as it can be inferred
        
        self.div_thr = 100

    @property
    def S(self):
        return self.dim

    def run(self, time, record_interval=None, **kwargs):
        
        if self._trans_string == 'log':
            div_thr = np.log(self.div_thr * self._K)
        else:
            div_thr = self.div_thr * self._K
        
        def divergence_event_tracker(t,y):
            nonlocal div_thr
            return np.max(y/div_thr) - 1

        divergence_event_tracker.terminal = True
        
        kwargs.update({'events' : divergence_event_tracker})
        return super(LotkaVolterra,self).run(time,record_interval,**kwargs)

    """
    Below are property code to support the alternative parameterization of LV
    """

    @property
    def mat_alpha(self):
        vec_K = self._K if type(self._K) == np.ndarray else np.full(self.dim, self._K)
        vec_r = self._r if type(self._r) == np.ndarray else np.full(self.dim, self._r)
        return -(vec_K / vec_r)[:,None] * self.mat_b / vec_K

    @mat_alpha.setter
    def mat_alpha(self, mat):
        vec_K = self._K if type(self._K) == np.ndarray else np.full(self.dim, self._K)
        vec_r = self._r if type(self._r) == np.ndarray else np.full(self.dim, self._r)
        self.mat_b = -(vec_r / vec_K)[:,None] * mat * vec_K

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val):
        mat_alpha = self.mat_alpha.copy()
        self._r = val
        self.mat_alpha = mat_alpha  # interaction mat needs to be rescaled
        self.k = self._r

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, val):
        mat_alpha = self.mat_alpha.copy()
        self._K = val
        self.mat_alpha = mat_alpha  # interaction mat needs to be rescaled
        self.c = np.array(self._lam / self._K)

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, val):
        self._lam = val
        self.c = np.array(val / self._K)
        
