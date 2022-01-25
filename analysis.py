import os
import glob
import numpy as np
import itertools
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap


inch = 3.14

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14


def generate_cmap(colors, num, name='custom_cmap'):
    _values = range(len(colors))
    _vmax = np.ceil(np.max(_values))
    _color_list = []
    for _v, _c in zip(_values, colors):
        _color_list.append((_v/_vmax, _c))
    _arg = name, _color_list, num
    return LinearSegmentedColormap.from_list(*_arg)


def generate_disc_cmap(colors, vals, name='custom_cmap'):
    cmap = ListedColormap(colors)
    bounds = vals
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def check_sample(chains, labels=None, colors=['dodgerblue', 'deeppink'], DoF=1,
                 burnin=0, thinning=1, show=True, output=None, **kwargs):
    _cm = generate_cmap(colors, len(chains))
    nbeta = chains[0].beta.shape[1]
    fig = plt.figure(figsize=(inch*3, inch*(nbeta+1)), facecolor='white')
    gs = fig.add_gridspec(nbeta+1, 5)
    ax = []
    ax_hist = []
    ax.append(fig.add_subplot(gs[nbeta, 0:4]))
    ax[0].tick_params(top=True, bottom=True,
                      left=True, right=True)
    for _cid, _c in enumerate(chains):
        ax[0].plot(_c.x2[burnin::thinning]/DoF,
                   color=_cm(_cid), label='Chain {:d}'.format(_cid))
        # ax[0].set_yscale('log')
    xmax = max([_c.x2[burnin::thinning].size-1 for _c in chains])
    x2_min = min([_c.x2.min() for _c in chains])/DoF
    ylabel = r'$\chi^2$' if DoF == 1 else r'$\chi^2_{red}$'
    ax[0].hlines(x2_min, 0, xmax, linestyle='dashed', color='k',
                 label=r'min({:s}) : {:.2f}'.format(ylabel, x2_min))
    ax[0].set_xlabel('steps')
    ax[0].set_ylabel(ylabel)
    ax[0].set_xlim(0, xmax+1)
    ax[0].legend(bbox_to_anchor=(1.26, 1.+nbeta),
                 loc='upper left', borderaxespad=0)
    for _betaid in range(nbeta):
        ax.append(fig.add_subplot(gs[_betaid, 0:4], sharex=ax[0]))
        ax_hist.append(fig.add_subplot(gs[_betaid, 4], sharey=ax[_betaid+1]))
        ax[_betaid+1].tick_params(top=True, bottom=True,
                                  left=True, right=True)
        ax_hist[_betaid].tick_params(top=False, bottom=False,
                                     left=True, right=False)
        plt.setp(ax[_betaid+1].get_xticklabels(), visible=False)
        plt.setp(ax_hist[_betaid].get_xticklabels(), visible=False)
        plt.setp(ax_hist[_betaid].get_yticklabels(), visible=False)
        for _cid, _c in enumerate(chains):
            ax[_betaid+1].plot(_c.beta[burnin::thinning, _betaid],
                               color=_cm(_cid),
                               label='Chain {:d}'.format(_cid))
            ax_hist[_betaid].hist(_c.beta[burnin::thinning, _betaid],
                                  color=_cm(_cid), orientation="horizontal",
                                  histtype='step', density=True)
        ylabel = r'coef[{:d}]'.format(_betaid) if labels is None \
            else labels[_betaid]
        ax[_betaid+1].set_ylabel(ylabel)
    fig.align_labels()
    plt.subplots_adjust(wspace=0., hspace=.02)
    if show:
        plt.show()
    if output is not None:
        fig.savefig(output, **kwargs)


def check_autocorrelation(chains, labels=None,
                          colors=['dodgerblue', 'deeppink'],
                          burnin=0, thinning=1, show=True, output=None,
                          **kwargs):
    maxlag = 100
    _cm = generate_cmap(colors, len(chains))
    nbeta = chains[0].beta.shape[1]
    fig = plt.figure(figsize=(inch*3, inch*(nbeta)), facecolor='white')
    gs = fig.add_gridspec(nbeta,)
    ax = []
    for _betaid in range(nbeta):
        if _betaid == 0:
            ax.append(fig.add_subplot(gs[_betaid, 0]))
            plt.setp(ax[_betaid].get_xticklabels(), visible=False)
            ax[_betaid].set_xlim(0, maxlag)
        elif _betaid == nbeta-1:
            ax.append(fig.add_subplot(gs[_betaid, 0], sharex=ax[0]))
            ax[_betaid].set_xlabel('lags')
        else:
            ax.append(fig.add_subplot(gs[_betaid, 0], sharex=ax[0]))
            plt.setp(ax[_betaid].get_xticklabels(), visible=False)
        ax[_betaid].tick_params(top=True, bottom=True,
                                left=True, right=True)
        for _cid, _c in enumerate(chains):
            seleas = pd.Series(_c.beta[burnin::thinning, _betaid])
            if _betaid == 0:
                autocorrelation_plot(seleas, ax=ax[_betaid], color=_cm(_cid),
                                     label='Chain {:d}'.format(_cid))
            else:
                autocorrelation_plot(seleas, ax=ax[_betaid], color=_cm(_cid))
            ax[_betaid].set_ylim(-0.7, 1.2)
        if _betaid == 0:
            ax[_betaid].legend(bbox_to_anchor=(1.01, 1.0),
                               loc='upper left', borderaxespad=0)
        ylabel = r'AutoCorr (coef[{:d}])'.format(_betaid) if labels is None \
            else r'AutoCorr ({:s})'.format(labels[_betaid])
        ax[_betaid].set_ylabel(ylabel)
    fig.align_labels()
    plt.subplots_adjust(hspace=.02)
    if show:
        plt.show()
    if output is not None:
        fig.savefig(output, **kwargs)


def check_hist(chains, labels=None, burnin=0, thinning=1, histtype='scatter',
               show=True, output=None, **kwargs):
    for _cid, _c in enumerate(chains):
        if _cid == 0:
            _res = np.array(_c.beta[burnin::thinning])
        else:
            _res = \
                np.concatenate([_res, np.array(_c.beta[burnin::thinning])])
    fig = plt.figure(figsize=(inch*_res.shape[1], inch*_res.shape[1]),
                     facecolor='white')
    gs = fig.add_gridspec(_res.shape[1], _res.shape[1])
    ax1d = list()
    ax2d = dict()
    nbin = list()
    for _i in range(_res.shape[1]):
        ax1d.append(fig.add_subplot(gs[_i, _i]))
        ax1d[_i].tick_params(top=False, bottom=True,
                             left=False, right=False)
        nbin.append(1+int(np.log2(_res.shape[0])))
        # nbin.append(30)
        ax1d[_i].hist(_res[:, _i], bins=nbin[_i], color='grey', alpha=0.3)
        if _i != _res.shape[1]-1:
            plt.setp(ax1d[_i].get_xticklabels(), visible=False)
        else:
            plt.setp(ax1d[_i].get_xticklabels(), rotation=45)
            xlabel = r'coef[{:d}]'.format(_i) if labels is None else labels[_i]
            ax1d[_i].set_xlabel(xlabel)
        x_min, x_max = ax1d[_i].get_xlim()
        dx = x_max - x_min
        ax1d[_i].set_xlim(x_min-dx*0.2, x_max+dx*0.2)
        plt.setp(ax1d[_i].get_yticklabels(), visible=False)
    _args = \
        itertools.combinations(range(_res.shape[1]), 2)
    for _i, _j in _args:
        _idx = (_i, _j)
        ax2d[_idx] = fig.add_subplot(gs[_j, _i])
        ax2d[_idx].tick_params(top=True, bottom=True,
                               left=True, right=True)
        if histtype == 'scatter':
            ax2d[_idx].scatter(_res[:, _i], _res[:, _j], color='grey', s=1)
        elif histtype in ['heatmap', 'contour']:
            x_min, x_max = ax1d[_i].get_xlim()
            y_min, y_max = ax1d[_j].get_xlim()
            x_bins = np.linspace(x_min, x_max, nbin[_i])
            y_bins = np.linspace(y_min, y_max, nbin[_j])
            hist, xe, ye = np.histogram2d(_res[:, _i], _res[:, _j],
                                          bins=(x_bins, y_bins))
            hist = hist/np.sum(hist)
            if histtype == 'heatmap':
                ax2d[_idx].imshow(hist.T, interpolation='nearest',
                                  origin='lower', aspect='auto',
                                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                                  cmap='Greys')

            elif histtype == 'contour':
                x = [np.mean(xe[_k:_k+2]) for _k in range(xe.size-1)]
                y = [np.mean(ye[_k:_k+2]) for _k in range(ye.size-1)]
                x, y = np.meshgrid(x, y)
                levels = [hist.max()]
                for lev in [0.682689492, 0.954499736, 0.997300204]:
                    hist_flatsort = \
                        np.sort(hist[np.where(hist > 0.)].flatten())
                    hist_sum = np.array([np.sum(hist_flatsort[0:i+1])
                                         for i in range(hist_flatsort.size)])
                    temp_idx = (np.abs(hist_sum-(1.-lev))).argmin()
                    levels.append(hist_flatsort[temp_idx])
                levels = sorted(list(set(levels)))
                cmap, norm = \
                    generate_disc_cmap(['whitesmoke', 'darkgray', 'k'],
                                       vals=levels)
                ax2d[_idx].contourf(x, y, hist.T, levels=levels,
                                    cmap=cmap, norm=norm,
                                    vmin=min(levels), vmax=max(levels))
                ax2d[_idx].contour(x, y, hist.T, levels=levels[0:-1],
                                   colors='k', linewidths=0.5)
        ax2d[_idx].set_xlim(ax1d[_i].get_xlim())
        ax2d[_idx].set_ylim(ax1d[_j].get_xlim())
        if _i == 0:
            ylabel = r'coef[{:d}]'.format(_j) if labels is None else labels[_j]
            ax2d[_idx].set_ylabel(ylabel)
        else:
            plt.setp(ax2d[_idx].get_yticklabels(), visible=False)
        if _j == _res.shape[1]-1:
            plt.setp(ax2d[_idx].get_xticklabels(), rotation=45)
            xlabel = r'coef[{:d}]'.format(_i) if labels is None else labels[_i]
            ax2d[_idx].set_xlabel(xlabel)
        else:
            plt.setp(ax2d[_idx].get_xticklabels(), visible=False)
    fig.align_labels()
    plt.subplots_adjust(hspace=.0, wspace=0.)
    if show:
        plt.show()
    if output is not None:
        fig.savefig(output, **kwargs)


def make_output(chains, burnin=0, thinning=1):
    for _cid, _c in enumerate(chains):
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
    return np.array(_mean), np.array(_std), _cov


def calc_range(coef, cov, xrange, function, nxdata=10, nsample=1000):
    _x = np.linspace(xrange[0], xrange[1], num=nxdata)
    _copy_cov = np.copy(cov)
    _cov_diag = np.diag(_copy_cov)
    _scale = np.nan_to_num(np.diag(1./_cov_diag))
    _scale_inv = np.diag(_cov_diag)
    _scaled_mean = np.sqrt(_scale)@coef
    _scaled_cov = np.matmul(_scale, cov)
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
    _y = np.array([function(_c, _x) for _c in _coef_list])
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


class _ChainClass:
    def __init__(self, _beta0, _Lh, _x2):
        self.beta = np.array(_beta0)
        self.Lh = np.array(_Lh)
        self.x2 = np.array(_x2)


def read_result(path, idxs=None):
    files = glob.glob(os.path.join(path, 'chain_*.dat'))
    nchain = len(files)
    chains = []
    for _cidx in range(nchain):
        filename = os.path.join(path, 'chain_{:0>2d}.dat'.format(_cidx))
        try:
            df = pd.read_csv(filename, sep=",")
            beta = [key for key in df.columns
                    if key not in ['Likelihood', 'chi2']] if idxs is None \
                else idxs
            chains.append(_ChainClass(df[beta], df.Likelihood, df.chi2))
        except pd.errors.EmptyDataError:
            continue
    return chains


# for main
if __name__ == '__main__':
    pass
