from .util import get_rotation_matrix
from pyrocko.gf.seismosizer import Source, SourceWithMagnitude
from pyrocko.gf import meta
from pyrocko.guts import Float
from pyrocko import moment_tensor as mtm

import numpy as num

import logging

pi = num.pi
pi4 = pi / 4.
km = 1000.
d2r = pi / 180.
r2d = 180. / pi

sqrt3 = num.sqrt(3.)
sqrt2 = num.sqrt(2.)
sqrt6 = num.sqrt(6.)

logger = logging.getLogger('sources')


class MTQTSource(SourceWithMagnitude):
    """
    A moment tensor point source.

    Notes
    -----
    Following Q-T parameterization after Tape & Tape 2015
    """

    discretized_source_class = meta.DiscretizedMTSource

    u = Float.T(
        default=0.,
        help='Lune co-latitude transformed to grid.'
             'Defined: 0 <= u <=3/4pi')

    v = Float.T(
        default=0.,
        help='Lune co-longitude transformed to grid.'
             'Definded: -1/3 <= v <= 1/3')

    kappa = Float.T(
        default=0.,
        help='Strike angle equivalent of moment tensor plane.'
             'Defined: 0 <= kappa <= 2pi')

    sigma = Float.T(
        default=0.,
        help='Rake angle equivalent of moment tensor slip angle.'
             'Defined: -pi/2 <= sigma <= pi/2')

    h = Float.T(
        default=0.,
        help='Dip angle equivalent of moment tensor plane.'
             'Defined: 0 <= h <= 1')

    def __init__(self, **kwargs):
        n = 1000
        self._beta_mapping = num.linspace(0, pi, n)
        self._u_mapping = \
            (3. / 4. * self._beta_mapping) - \
            (1. / 2. * num.sin(2. * self._beta_mapping)) + \
            (1. / 16. * num.sin(4. * self._beta_mapping))

        self.lambda_factor_matrix = num.array(
            [[sqrt3, -1., sqrt2],
             [0., 2., sqrt2],
             [-sqrt3, -1., sqrt2]], dtype='float64')

        self.R = get_rotation_matrix()
        self.roty_pi4 = self.R['y'](-pi4)
        self.rotx_pi = self.R['x'](pi)

        self._lune_lambda_matrix = num.zeros((3, 3), dtype='float64')

        Source.__init__(self, **kwargs)

    @property
    def gamma(self):
        """
        Lunar co-longitude, dependend on v
        """
        return (1. / 3.) * num.arcsin(3. * self.v)

    @property
    def beta(self):
        """
        Lunar co-latitude, dependend on u
        """
        return num.interp(self.u, self._u_mapping, self._beta_mapping)

    def delta(self):
        """
        From Tape & Tape 2012, delta measures departure of MT being DC
        Delta = Gamma = 0 yields pure DC
        """
        return (pi / 2.) - self.beta

    @property
    def theta(self):
        return num.arccos(self.h)

    @property
    def rot_theta(self):
        return self.R['x'](self.theta)

    @property
    def rot_kappa(self):
        return self.R['z'](-self.kappa)

    @property
    def rot_sigma(self):
        return self.R['z'](self.sigma)

    @property
    def lune_lambda(self):
        sin_beta = num.sin(self.beta)
        cos_beta = num.cos(self.beta)
        sin_gamma = num.sin(self.gamma)
        cos_gamma = num.cos(self.gamma)
        vec = num.array([sin_beta * cos_gamma, sin_beta * sin_gamma, cos_beta])
        return 1. / sqrt6 * self.lambda_factor_matrix.dot(vec)

    @property
    def lune_lambda_matrix(self):
        num.fill_diagonal(self._lune_lambda_matrix, self.lune_lambda)
        return self._lune_lambda_matrix

    @property
    def rot_V(self):
        return self.rot_kappa.dot(self.rot_theta).dot(self.rot_sigma)

    @property
    def rot_U(self):
        return self.rot_V.dot(self.roty_pi4)

    @property
    def m9_nwu(self):
        """
        MT orientation is in NWU
        """
        return self.rot_U.dot(
            self.lune_lambda_matrix).dot(num.linalg.inv(self.rot_U))

    @property
    def m9(self):
        """
        Pyrocko MT in NED
        """
        return self.rotx_pi.dot(self.m9_nwu).dot(self.rotx_pi.T)

    @property
    def m6(self):
        return mtm.to6(self.m9)

    @property
    def m6_astuple(self):
        return tuple(self.m6.ravel().tolist())

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, 0.0)
        m0 = mtm.magnitude_to_moment(self.magnitude)
        m6s = self.m6 * m0
        return meta.DiscretizedMTSource(
            m6s=m6s[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times))

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            d.update(m6=list(map(float, mt.m6())))

        d.update(kwargs)
        return super(MTQTSource, cls).from_pyrocko_event(ev, **d)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['R'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.R = get_rotation_matrix()