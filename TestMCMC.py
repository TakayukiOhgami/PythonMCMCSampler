from mkTestData import mkdata
from fittingclass import PriorClass, MetropolisAlgorithm, chi2


testdata = mkdata('q', [1.5, 0, 2], [-15, 15], 15, 0, 10, 'return', 'random')
testdata.xsort()

prior = PriorClass([1, 0, 1], std=[0.1, 0.1, 0.1], cov=None)
mcmc = MetropolisAlgorithm(testdata, function='q', prior=prior)
mcmc.setChain(1000, nchain=2, stepsize=0.3, betastep=[1., 0., 1.])
mcmc.run(prior=False, outdir='output', multi=True)
output = mcmc.make_output(burnin=100, thinning=10)

print(output.coef)
print(output.coef_std)
print(output.coef_cov)
redchi2 = 'reduced chi squared: {:.1E}'.format(chi2(testdata, mcmc.function(output.coef, testdata.data['test'].x))/mcmc.DoF)
print(redchi2)
