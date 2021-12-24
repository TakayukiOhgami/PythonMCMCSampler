import numpy as np
import matplotlib.pyplot as plt
from library.dataclass import DataClass
from library.fittingclass import PriorClass, MetropolisAlgorithm

def readDataFile(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
    x, y, xerr, yerr = [], [], [], []
    for l in lines:
        infos = l.replace('\n', '').split(' ')
        x.append(float(infos[0]))
        y.append(float(infos[1]))
        xerr.append(float(infos[2]))
        yerr.append(float(infos[3]))
    return DataClass(name='data', x=x, y=y, xerr=xerr, yerr=yerr)

data = readDataFile('testdata.dat')
data.sort()


prior = PriorClass([0., 0., 0.], std=[0.05, 0.05, 0.5])

mcmc = MetropolisAlgorithm(data, function='q', prior=prior)
mcmc.setChain(20000, nchain=1, stepsize=0.3, betastep=[1., 1., 1.])
mcmc.run(prior=False, outdir='output')

output = mcmc.make_output(burnin=5000, thinning=10)

print(output.coef)
print(output.coef_std)
print(output.coef_cov)
