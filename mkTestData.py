import sys
import numpy as np
import PythonMCMCSampler.dataclass as dc
from argparse import ArgumentParser
from PythonMCMCSampler.fittingclass import linear_func
from PythonMCMCSampler.fittingclass import quadratic_func
from PythonMCMCSampler.fittingclass import poly_function
from PythonMCMCSampler.fittingclass import gaussian_func
from PythonMCMCSampler.fittingclass import normal_dist_func


def mkdata(function, coef, rx, nx=100, vx=0, vy=0,
           output='./testdata.dat', mode='random'):
    if function in [1, '1', 'l', 'L', 'linear']:
        func = linear_func
    elif function in [2, '2', 'q', 'Q', 'quadratic']:
        func = quadratic_func
    elif function in ['p', 'P', 'polynomial', 'polyval']:
        func = poly_function
    elif function in ['n', 'N', 'normal', 'Normal']:
        func = normal_dist_func
    elif function in ['g', 'G', 'gaussian', 'Gaussian']:
        func = gaussian_func
    else:
        _e = 'Please choose from linear, quadratic, poly, normal or gaussian'
        sys.stderr.write(_e)

    if mode == 'random':
        x_array = np.random.uniform(low=rx[0],
                                    high=rx[-1],
                                    size=nx)
    else:
        x_array = np.linspace(rx[0], rx[-1], nx)
    y_array = func(coef, x_array)
    if mode == 'random':
        x_array = x_array + np.random.normal(loc=0., scale=vx,
                                             size=nx)
        y_array = y_array + np.random.normal(loc=0., scale=vy,
                                             size=nx)
    xerr_array = np.full(nx, vx)
    yerr_array = np.full(nx, vy)

    _ret = (x_array, y_array, xerr_array, yerr_array)
    if output == 'return':
        _dataclass =\
            dc.DataClass(name='test',
                         x=_ret[0], y=_ret[1], xerr=_ret[2], yerr=_ret[3])
        return _dataclass
    else:
        print('x,y,xerr,yerr')
        with open(output, 'w') as fout:
            for x, y, xerr, yerr in zip(*_ret):
                print(x, y, xerr, yerr, sep=',')
                print(x, y, xerr, yerr, file=fout, sep=',')


def _get_option():
    _usage = 'Usage: python {} [--help]'.format(__file__)
    _usage += ' function <str>'
    _usage += ' --coefficient <float list>'
    _usage += ' --rx <float list>'
    _usage += ' --nx <int>'
    _usage += ' --vx <float>'
    _usage += ' --vy <float>'
    _usage += ' --output <str>'
    _argparser = ArgumentParser(usage=_usage)
    _argparser.add_argument('function',
                            type=str,
                            help='[linear, quadratic, poly,'
                                 ' normal or gaussian]')
    _argparser.add_argument('-c', '--coef',
                            required=True, nargs='*', type=float,
                            help='List of coefficient')
    _argparser.add_argument('-r', '--rx',
                            required=True, nargs='*', type=float,
                            help='Range of x [xmin, xmax]')
    _argparser.add_argument('--nx',
                            type=int, default=100,
                            help='Number of x <default 100>')
    _argparser.add_argument('--vx',
                            type=float, default=0,
                            help='Variance of x <default 0>')
    _argparser.add_argument('--vy',
                            type=float, default=0,
                            help='Variance of y <default 0>')
    _argparser.add_argument('-o', '--output',
                            type=str, default='testdata.dat',
                            help='Output file name'
                                 ' <default ./testdata.dat>')
    return _argparser.parse_args()


if __name__ == '__main__':
    optargs = _get_option()
    mkdata(optargs.function, optargs.coef,
           optargs.rx, optargs.nx,
           optargs.vx, optargs.vy,
           optargs.output)
