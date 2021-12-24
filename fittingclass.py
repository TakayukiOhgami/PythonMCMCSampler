import os
import sys
import math
import numpy as np
import scipy.odr as ODR
from scipy.stats import norm, multivariate_normal
from numpy.linalg import LinAlgError
import dataclass as dc
from random import random
from random import uniform
from random import sample
from mydecorator import stop_watch, log, get_logger
from mpi4py import MPI

logger = get_logger()

_comm = MPI.COMM_WORLD
_myrank = _comm.Get_rank()
_numcores = _comm.Get_size()


def linear_func(coef, x):
    return coef[0] * x + coef[1]


def quadratic_func(coef, x):
    return coef[0] * np.square(x) + coef[1] * x + coef[2]


def poly_function(coef, x):
    return np.polyval(coef, x)


def normal_dist_func(coef, x):
    _a = 1./(np.sqrt(2.*np.pi)*coef[1])
    return _a * np.exp(-.5*np.square((x - coef[0])/coef[1]))


def gaussian_func(coef, x):
    return coef[0] * np.exp(-.5*np.square((x - coef[1])/coef[2]))


def chi2(obsdata, template, dataname=None):
    if type(template) is dc.DataClass:
        x2 = 0.
        names = obsdata.name() if dataname is None else [dataname]
        for name in names:
            tags = [None] if template.data[name].tag is None \
                else np.unique(template.data[name].tag)
            for idx, tag in enumerate(tags):
                tod = dc.pick(obsdata, name=name, tag=tag)
                ttd = dc.pick(template, name=name, tag=tag)
                index = np.array([np.where(ttd.data[name].x == _v)
                                  for _v in tod.data[name].x]).flatten()
                if tod.data[name].z is None:
                    v, verr = tod.data[name].y, tod.data[name].yerr
                    v_t = ttd.data[name].y[index]
                    x2 += np.sum(np.square((v-v_t)/verr))
                else:
                    y, yerr = tod.data[name].y, tod.data[name].yerr
                    z, zerr = tod.data[name].z, tod.data[name].zerr
                    y_t = ttd.data[name].y[index]
                    z_t = ttd.data[name].z[index]
                    x2 += np.sum(np.square((y-y_t)/yerr))
                    x2 += np.sum(np.square((z-z_t)/zerr))
    else:
        name = obsdata.name()[0] if dataname is None else dataname
        v, verr = obsdata.data[name].y, obsdata.data[name].yerr
        x2 = np.sum(np.square((v-template)/verr))
    return x2


def log_likelihood(obsdata, v_temp, prior=1., dataname=None):
    x2 = chi2(obsdata, v_temp, dataname=dataname)
    lh = -.5*x2 + math.log(prior)
    return lh


class FittingClassBase:
    def __init__(self, data, function):
        if not isinstance(data, dc.DataClass):
            sys.stderr.write('Please use dataclass.DataClass for data.')
        else:
            self.data = data
        if function in [1, '1', 'l', 'L', 'linear']:
            self.function = linear_func
        elif function in [2, '2', 'q', 'Q', 'quadratic']:
            self.function = quadratic_func
        elif function in ['p', 'P', 'polynomial', 'polyval']:
            self.function = poly_function
        elif function in ['n', 'N', 'normal', 'Normal']:
            self.function = normal_dist_func
        elif function in ['g', 'G', 'gaussian', 'Gaussian']:
            self.function = gaussian_func
        else:
            self.function = function

    class _OutputClass:
        def __init__(self, coef, coef_std, coef_cov):
            self.coef = coef
            self.coef_std = coef_std
            self.coef_cov = coef_cov

    @log(logger)
    def calc_range(self, xrange, nxdata=10, nsample=1000):
        _x = np.linspace(xrange[0], xrange[1], num=nxdata)
        _copy_cov = np.copy(self.output.coef_cov)
        _cov_diag = np.diag(_copy_cov)
        _scale = np.nan_to_num(np.diag(1./_cov_diag))
        _scale_inv = np.diag(_cov_diag)
        _scaled_mean = np.sqrt(_scale)@self.output.coef
        _scaled_cov = np.matmul(_scale, self.output.coef_cov)
        try:
            _mvn = multivariate_normal(mean=_scaled_mean, cov=_scaled_cov)
        except LinAlgError:
            _copy_cov = np.copy(_scaled_cov)
            _cov_diag = np.diag(_copy_cov)
            _cov_diag.flags.writeable = True
            _cov_diag += float(_cov_diag[_cov_diag != 0.].min()*1.E-5)
            _mvn = multivariate_normal(mean=_scaled_mean, cov=_copy_cov)
        _coef_list = [np.sqrt(_scale_inv)@_sample
                      for _sample in _mvn.rvs(size=nsample)]
        _y = np.array([self.function(_c, _x) for _c in _coef_list])
        if _y.ndim == 2:
            _sample = [(np.mean(_y.T[_i, :]), np.std(_y.T[_i, :]))
                       for _i in range(nxdata)]
            _maxval = [_m+_s for _m, _s in _sample]
            _minval = [_m-_s for _m, _s in _sample]
        else:
            _maxval = list()
            _minval = list()
            for _idx in range(_y.shape[1]):
                _sample = [(np.mean(_y.T[_i, _idx, :]),
                            np.std(_y.T[_i, _idx, :]))
                           for _i in range(nxdata)]
                _maxval.append([_m+_s for _m, _s in _sample])
                _minval.append([_m-_s for _m, _s in _sample])
        return np.array(_x), np.array(_maxval), np.array(_minval)


class ODRFittingClass(FittingClassBase):
    def __init__(self, data, function, beta0, fit_type=0, delta0=None):
        super(ODRFittingClass, self).__init__(data, function)
        _sx = False if data.data.xerr is None else data.data.xerr
        _sy = False if data.data.yerr is None else data.data.yerr
        self.ODRRealData = ODR.RealData(x=data.data.x, y=data.data.y,
                                        sx=_sx, sy=_sy)
        self.ODRModel = ODR.Model(self.function)
        self.ODR = ODR.ODR(data=self.ODRRealData, model=self.ODRModel,
                           beta0=beta0, delta0=delta0)
        self.ODR.set_job(fit_type=fit_type)

    def run(self):
        output = self.ODR.run()
        self.output = self._OutputClass(output.beta,
                                        output.sd_beta,
                                        output.cov_beta)
        return self.output


class PriorClass:
    @log(logger)
    def __init__(self, mean, std=None, cov=None):
        if isinstance(mean, (int, float)):
            self.mean = float(mean)
            self.std = math.sqrt(self.mean) if std is None else float(std)
            self._func = norm(loc=self.mean, scale=self.std)
            self.scale = np.diag(1./(self.std*self.std))
            self.scale_inv = np.diag(self.std*self.std)
        else:
            self.mean = np.array(mean)
            if std is None and cov is None:
                sys.stderr.write('Please set std or cov.')
            elif std is None and cov is not None:
                self.cov = np.array(cov)
                self.std = np.sqrt(self.cov.diagonal())
            elif std is not None and cov is None:
                self.std = np.array(std)
                self.cov = np.diag(self.std*self.std)
            else:
                self.std = np.array(std)
                self.cov = np.array(cov)
            self.scale = np.nan_to_num(np.diag(1./(self.std*self.std)))
            self.scale_inv = np.diag(self.std*self.std)
            self.scaled_mean = np.sqrt(self.scale)@self.mean
            self.scaled_cov = np.matmul(self.scale, self.cov)
            try:
                self._func = multivariate_normal(mean=self.scaled_mean,
                                                 cov=self.scaled_cov)
            except LinAlgError:
                logger.warning('Add a small diagonal matrix.')
                _copy_cov = np.copy(self.scaled_cov)
                _cov_diag = np.diag(_copy_cov)
                _cov_diag.flags.writeable = True
                _cov_diag += float(_cov_diag[_cov_diag != 0.].min()*1.E-5)
                self._func = multivariate_normal(mean=self.scaled_mean,
                                                 cov=_copy_cov)

    def pdf(self, pos):
        _p0 = self._func.pdf(np.sqrt(self.scale)@self.mean)
        return self._func.pdf(np.sqrt(self.scale)@pos)/_p0

    def rvs(self, size=1):
        if size == 1:
            return np.sqrt(self.scale_inv)@self._func.rvs(size=size)
        else:
            return [np.sqrt(self.scale_inv)@_sample
                    for _sample in self._func.rvs(size=size)[:]]

    def initial(self, vrange=5., size=1):
        _ret = []
        for _i in range(size):
            _ret.append(np.array([uniform(_m-vrange*_s, _m+vrange*_s)
                                  for _m, _s in zip(self.mean, self.std)]))
        return _ret


def _unwrap_MCMC(arg, **kwarg):
    return MCMCFittingClassBase._wrapper(*arg)


class MCMCFittingClassBase(FittingClassBase):
    def __init__(self, data, function, prior):
        super(MCMCFittingClassBase, self).__init__(data, function)
        if not isinstance(prior, PriorClass):
            sys.stderr.write('Please use PriorClass for prior.')
        else:
            self.prior = prior

    @log(logger)
    def setChain(self, length, nchain=1, beta0=None, stepsize=1., betastep=1.):
        self.nchain = int(nchain)
        self.length = int(length)
        self.stepsize = np.full(self.nchain, stepsize) \
            if type(stepsize) is float else np.array(stepsize)
        self.betastep = np.full(len(self.prior.mean), betastep) \
            if type(betastep) is float else np.array(betastep)
        _datasize = 0
        for _data in self.data.data.values():
            _datasize += len(_data.y)
            if _data.z is not None:
                _datasize += len(_data.z)
        self.DoF = _datasize-np.where(self.betastep != 0)[0].size
        if beta0 is None:
            if nchain > 1:
                if _myrank == 0:
                    logger.info("Seting initial randomly.")
                    _b0_list = self.prior.initial(size=int(nchain))[:]
                    for _idx in np.where(self.betastep == 0.):
                        for _b0 in _b0_list:
                            _b0[_idx] = self.prior.mean[_idx]
                    self.chains = [self._ChainClass(*self._sampling(_b0)[0:3])
                                   for _b0 in list(_b0_list)]
                else:
                    self.chains = None
                self.chains = _comm.bcast(self.chains, root=0)
            else:
                logger.info("Seting initial to prior's mean.")
                _b0 = self.prior.mean
                for _idx in np.where(self.betastep == 0.):
                    _b0[_idx] = self.prior.mean[_idx]
                self.chains = [self._ChainClass(*self._sampling(_b0)[0:3])]
        else:
            logger.info("Seting initial.")
            self.chains = [self._ChainClass(*self._sampling(_b0)[0:3])
                           for _b0 in beta0]

    class _ChainClass:
        def __init__(self, _beta0, _Lh, _x2):
            self.beta = np.array([_beta0])
            self.Lh = np.array([float(_Lh)])
            self.x2 = np.array([float(_x2)])

        def add(self, _beta, _Lh, _x2):
            self.beta = np.append(self.beta, [_beta], axis=0)
            self.Lh = np.append(self.Lh, float(_Lh))
            self.x2 = np.append(self.x2, float(_x2))

    @log(logger)
    def make_output(self, burnin=0, thinning=1):
        for _cid, _c in enumerate(self.chains):
            if _cid == 0:
                _res = np.array(_c.beta[burnin::thinning])
            else:
                _res = \
                    np.concatenate([_res, np.array(_c.beta[burnin::thinning])])
        _mean, _std = [], []
        for _idx in range(_res.shape[1]):
            _mean.append(np.mean(_res[:, _idx]))
            _std.append(np.std(_res[:, _idx]))
        _cov = np.cov(_res.T)
        self.output = self._OutputClass(np.array(_mean), np.array(_std), _cov)
        return self.output


class MetropolisAlgorithm(MCMCFittingClassBase):
    def __init__(self, *arg, **kwargs):
        super(MetropolisAlgorithm, self).__init__(*arg, **kwargs)

    @log(logger)
    def _sampling(self, pre_beta, pre_Lh=None, prior=False, stepsize=1.):
        _x = np.unique(np.hstack([dc.pick(self.data, name=name).data[name].x
                                  for name in self.data.name()]))
        stepsize = 1. if pre_Lh is None else stepsize
        if isinstance(self.prior.mean, float):
            _std = self.prior.std*stepsize*self.betastep
            _suggest_pdf = norm(loc=pre_beta, scale=_std)
        else:
            _step_tensor = np.outer(self.betastep, self.betastep)*(stepsize**2)
            _cov = self.prior.cov*_step_tensor
            _scaled_mean = np.sqrt(self.prior.scale)@pre_beta
            _scaled_cov = np.matmul(self.prior.scale, _cov)
            try:
                _suggest_pdf = multivariate_normal(mean=_scaled_mean,
                                                   cov=_scaled_cov)
            except LinAlgError:
                logger.warning('Add a small diagonal matrix.')
                _copy_cov = np.copy(_scaled_cov)
                _cov_diag = np.diag(_copy_cov)
                _cov_diag.flags.writeable = True
                _cov_diag += float(_cov_diag[_cov_diag != 0.].min()*1.E-5)
                _suggest_pdf = multivariate_normal(mean=_scaled_mean,
                                                   cov=_copy_cov)
        _ratio = 0.
        while _ratio < random():
            _suggest_beta = pre_beta if pre_Lh is None \
                else np.sqrt(self.prior.scale_inv)@_suggest_pdf.rvs(1)
            for _idx in np.where(self.betastep == 0.):
                _suggest_beta[_idx] = self.prior.mean[_idx]
            _y_suggest = self.function(_suggest_beta, _x)
            _prior = self.prior.pdf(_suggest_beta) if prior else 1.
            _suggest_x2 = chi2(self.data, _y_suggest)
            _suggest_Lh = log_likelihood(self.data, _y_suggest, prior=_prior)
            try:
                _ratio = 1. if pre_Lh is None else math.exp(_suggest_Lh-pre_Lh)
            except OverflowError:
                _ratio = 1. if _suggest_Lh > pre_Lh else 0.
        _suggest_redx2 = _suggest_x2/self.DoF
        _info = 'CHI2RED: {:.3f}'.format(_suggest_redx2) + \
            '    DoF:' + str(self.DoF) + '    SAMPLE:'
        for _sb in _suggest_beta:
            _info += ' '+str(_sb)
        logger.info(_info)
        return _suggest_beta, _suggest_Lh, _suggest_x2

    def _wrapper(self, _arg):
        _beta, _Lh, _x2, _prior, _size, _chain, _outputpath = _arg
        _output = open(_outputpath, 'w')
        _out_txt = \
            [str(_v) for _v in [*range(len(_beta)), 'Likelihood', 'chi2']]
        _output.write(' '.join(_out_txt)+'\r\n')
        _out_txt = \
            [str(_v) for _v in [*_beta, _Lh, _x2]]
        _output.write(' '.join(_out_txt)+'\r\n')
        for _i in range(self.length-1):
            _beta, _Lh, _x2 = self._sampling(_beta, _Lh,
                                             prior=_prior,
                                             stepsize=_size)
            _chain.add(_beta, _Lh, _x2)
            _out_txt = [str(_v) for _v in [*_beta, _Lh, _x2]]
            _output.write(' '.join(_out_txt)+'\r\n')
        _output.close()

    @stop_watch
    @log(logger)
    def run(self, prior=False, outdir='samples', multi=False):
        if not multi:
            if not os.path.exists(outdir):
                logger.info('mkdir -p %s' % outdir)
                os.system('mkdir -p %s' % outdir)
            else:
                logger.warning('Directory [%s] is already exists.' % outdir)
            _temp_name = 'chain_{idx:0>2}.dat'
            _output_path = \
                [os.path.join(outdir, _temp_name.format(**locals()))
                 for idx in range(self.nchain)]
            _ = [self._wrapper((_chain.beta[-1], _chain.Lh[-1],
                                _chain.x2[-1], prior, self.stepsize[_i],
                                _chain, _output_path[_i]))
                 for _i, _chain in enumerate(self.chains)]
        else:
            if not os.path.exists(outdir):
                logger.info('mkdir -p %s' % outdir)
                os.system('mkdir -p %s' % outdir)
            else:
                logger.warning('Directory [%s] is already exists.' % outdir)
            _temp_name = 'chain_{idx:0>2}.dat'
            _output_path = \
                [os.path.join(outdir, _temp_name.format(**locals()))
                 for idx in range(self.nchain)]
            _args = [(_chain.beta[-1], _chain.Lh[-1],
                      _chain.x2[-1], prior, self.stepsize[_i],
                      _chain, _output_path[_i])
                     for _i, _chain in enumerate(self.chains)]
            _cid_num_list = \
                [(self.nchain+_i)//_numcores for _i in range(_numcores)]
            _cid_list = [list(range(sum(_cid_num_list[:_rank]),
                                    sum(_cid_num_list[:_rank+1])))
                         for _rank in range(_numcores)]
            _comm.Barrier()
            _ = [self._wrapper(_args[_cid]) for _cid in _cid_list[_myrank]]
            _comm.Barrier()
            for _rank, _cl in enumerate(_cid_list):
                for _cid in _cl:
                    self.chains[_cid] \
                        = _comm.bcast(self.chains[_cid], root=_rank)
            else:
                pass
            _comm.Barrier()
            if _myrank == 0:
                pass
            else:
                sys.exit()


class MetropolisAlgorithm2(MetropolisAlgorithm):
    def __init__(self, *arg, **kwargs):
        super(MetropolisAlgorithm2, self).__init__(*arg, **kwargs)

    @log(logger)
    def _sampling(self, pre_beta, pre_Lh=None, prior=False, stepsize=1.):
        _x = np.unique(np.hstack([dc.pick(self.data, name=name).data[name].x
                                  for name in self.data.name()]))
        stepsize = 1. if pre_Lh is None else stepsize
        _suggest_beta = np.copy(pre_beta)
        if pre_Lh is None:
            _y_suggest = self.function(_suggest_beta, _x)
            _prior = self.prior.pdf(_suggest_beta) if prior else 1.
            _suggest_x2 = chi2(self.data, _y_suggest)
            _suggest_Lh = log_likelihood(self.data, _y_suggest, prior=_prior)
        else:
            temp_list = list(np.where(self.betastep != 0.)[0])
            for _idx in sample(temp_list, len(temp_list)):
                _info = 'Suggest parameter index: %i' % _idx
                logger.info(_info)
                _ratio = 0.
                while _ratio < random():
                    _std = self.prior.std[_idx]*stepsize*self.betastep[_idx]
                    _suggest_pdf = norm(loc=pre_beta[_idx], scale=_std)
                    _suggest_beta[_idx] = _suggest_pdf.rvs(1)
                    _y_suggest = self.function(_suggest_beta, _x)
                    _prior = self.prior.pdf(_suggest_beta) if prior else 1.
                    _suggest_x2 = chi2(self.data, _y_suggest)
                    _suggest_Lh = \
                        log_likelihood(self.data, _y_suggest, prior=_prior)
                    try:
                        _ratio = math.exp(_suggest_Lh-pre_Lh)
                    except OverflowError:
                        _ratio = 1. if _suggest_Lh > pre_Lh else 0.
        _suggest_redx2 = _suggest_x2/self.DoF
        _info = 'CHI2RED: {:.3f}'.format(_suggest_redx2) + \
            '    DoF:' + str(self.DoF) + '    SAMPLE:'
        for _sb in _suggest_beta:
            _info += ' '+str(_sb)
        logger.info(_info)
        return _suggest_beta, _suggest_Lh, _suggest_x2


# for main
if __name__ == '__main__':
    pass
