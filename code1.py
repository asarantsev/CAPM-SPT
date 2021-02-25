import pandas
import numpy
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from statsmodels.api import OLS

def convert(x):
    return numpy.log(1 + x/100)

# Preliminary Analysis
PriceDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'PriceReturns')
TotalDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'TotalReturns')
SizeDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'Size')
BillDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'Bill')
priceRet = PriceDF.values[:, 3:]
totalRet = TotalDF.values[:, 3:]
size = SizeDF.values[:, 3:]
priceRet = priceRet[:, ::-1]
totalRet = totalRet[:, ::-1]
size = size[:, ::-1]
bill = BillDF.values[:, 1]
RiskFree = convert(bill/12)
lsize = numpy.log(size)
N = 8
TMONTHS = (2020 - 1926) * 12
NMONTHS = 24
T = int(TMONTHS/NMONTHS)
Price = convert(priceRet)
Total = convert(totalRet)
empiricalBetaPrice = numpy.array([])
empiricalBetaTotal = numpy.array([])
weightList = numpy.array([])

# Computation of Empirical Beta
for k in range(N-1):
    for year in range(T):
        returnsPrice = Price[NMONTHS*year:NMONTHS*(year+1), k]
        benchmarkPrice = Price[NMONTHS*year:NMONTHS*(year+1), N-1]
        returnsTotal = Total[NMONTHS*year:NMONTHS*(year+1), k] - RiskFree[NMONTHS*year:NMONTHS*(year+1)]
        benchmarkTotal = Total[NMONTHS*year:NMONTHS*(year+1), N-1] - RiskFree[NMONTHS*year:NMONTHS*(year+1)]
        regPrice = scipy.stats.linregress(benchmarkPrice, returnsPrice)
        regTotal = scipy.stats.linregress(benchmarkTotal, returnsTotal)
        betaPrice = regPrice.slope - 1
        betaTotal = regTotal.slope - 1
        empiricalBetaPrice = numpy.append(empiricalBetaPrice, betaPrice)
        empiricalBetaTotal = numpy.append(empiricalBetaTotal, betaTotal)
    weightList = numpy.append(weightList, lsize[::NMONTHS, N-1] - lsize[::NMONTHS, k])

# Empirical Beta vs Weights Analysis, Mean Trend
pyplot.plot(weightList, empiricalBetaPrice, 'go')
pyplot.title('beta vs weights, price')
pyplot.show()
normBetaPrice = empiricalBetaPrice/abs(weightList)
pyplot.plot(weightList, normBetaPrice, 'go')
pyplot.title('normalized beta vs weights, price')
pyplot.show()
beta0 = numpy.mean(normBetaPrice)
print('beta = ', beta0)

# Empirical Beta vs Weights Analysis, Mean Trend
pyplot.plot(weightList, empiricalBetaTotal, 'go')
pyplot.title('beta vs weights, premia')
pyplot.show()
normBetaTotal = empiricalBetaTotal/abs(weightList)
pyplot.plot(weightList, normBetaTotal, 'go')
pyplot.title('normalized beta vs weights, premia')
pyplot.show()
beta1 = numpy.mean(normBetaTotal)
print('beta = ', beta1)

PriceReturns = numpy.array([])
PriceBenchmark = numpy.array([])
TotalReturns = numpy.array([])
TotalBenchmark = numpy.array([])

# Creation of Benchmark Top Decile Returns and Other Decile Returns
for k in range(N-1):
    PriceReturns = numpy.append(PriceReturns, [sum(Price[NMONTHS*year:NMONTHS*(year+1), k]) for year in range(T)])
    PriceBenchmark = numpy.append(PriceBenchmark, [sum(Price[NMONTHS*year:NMONTHS*(year+1), N-1]) for year in range(T)])
    TotalReturns = numpy.append(TotalReturns, [sum(Total[NMONTHS*year:NMONTHS*(year+1), k]) for year in range(T)])
    TotalBenchmark = numpy.append(TotalBenchmark, [sum(Total[NMONTHS*year:NMONTHS*(year+1), N-1]) for year in range(T)])

print('analysis for price returns')
NewBenchmark = numpy.array([abs(weightList[item]) * PriceBenchmark[item] for item in range(T*(N-1))])
residAlphaPrice = PriceReturns - beta0 * NewBenchmark - PriceBenchmark
pyplot.plot(weightList, residAlphaPrice, 'go')
pyplot.title('unsystematic risk vs weights, prices')
pyplot.show()
normResidPrice = residAlphaPrice/abs(weightList)**0.5
pyplot.plot(weightList, normResidPrice, 'go')
pyplot.title('normalized unsystematic risk vs weights, prices')
pyplot.show()
qqplot(normResidPrice, line = 's')
pyplot.title('normalized unsystematic risk, prices')
pyplot.show()
plot_acf(normResidPrice)
pyplot.title('normalized unsystematic risk, prices')
pyplot.show()
print('Shapiro-Wilk p = ', scipy.stats.shapiro(normResidPrice)[1])
print('Jarque-Bera p = ', scipy.stats.jarque_bera(normResidPrice)[1])
print('normalized unsystematic risk mean, stdev = ', numpy.mean(normResidPrice), numpy.std(normResidPrice))

print('Final Prices Regression')
NormalizedPrice = (PriceReturns - PriceBenchmark)/abs(weightList)**0.5
NewBenchmark = numpy.array([abs(weightList[item]) * PriceBenchmark[item] for item in range(T*(N-1))])
Reg = scipy.stats.linregress(NewBenchmark, NormalizedPrice)
print(Reg)
Res = NormalizedPrice - Reg.slope * NewBenchmark - Reg.intercept * numpy.ones(T*(N-1))
qqplot(Res, line = 's')
pyplot.title('Final Regression, Prices')
pyplot.show()
plot_acf(Res)
pyplot.title('Final Regression, Prices')
pyplot.show()
print('Shapiro-Wilk p = ', scipy.stats.shapiro(Res)[1])
print('Jarque-Bera p = ', scipy.stats.jarque_bera(Res)[1])
print('sigma = ', numpy.std(Res))

print('analysis for equity premia')
NewBenchmark = numpy.array([abs(weightList[item]) * TotalBenchmark[item] for item in range(T*(N-1))])
residAlphaTotal = TotalReturns - beta1 * NewBenchmark - TotalBenchmark
pyplot.plot(weightList, residAlphaTotal, 'go')
pyplot.title('unsystematic risk vs weights, premia')
pyplot.show()
normResidTotal = residAlphaTotal/abs(weightList)**0.5
pyplot.plot(weightList, normResidTotal, 'go')
pyplot.title('normalized unsystematic risk vs weights, premia')
pyplot.show()
qqplot(normResidTotal, line = 's')
pyplot.title('normalized unsystematic risk, premia')
pyplot.show()
plot_acf(normResidTotal)
pyplot.title('normalized unsystematic risk, premia')
pyplot.show()
print('Shapiro-Wilk p = ', scipy.stats.shapiro(normResidTotal)[1])
print('Jarque-Bera p = ', scipy.stats.jarque_bera(normResidTotal)[1])
print('normalized unsystematic risk mean, stdev = ', numpy.mean(normResidTotal), numpy.std(normResidTotal))

print('Final Total Regression')
NormalizedTotal = (TotalReturns - TotalBenchmark)/abs(weightList)**0.5
NewBenchmark = numpy.array([abs(weightList[item]) * TotalBenchmark[item] for item in range(T*(N-1))])
Reg = scipy.stats.linregress(NewBenchmark, NormalizedTotal)
print(Reg)
Res = NormalizedTotal - Reg.slope * NewBenchmark - Reg.intercept * numpy.ones(T*(N-1))
qqplot(Res, line = 's')
pyplot.title('Final Regression, Premia')
pyplot.show()
plot_acf(Res)
pyplot.title('Final Regression, Premia')
pyplot.show()
print('Shapiro-Wilk p = ', scipy.stats.shapiro(Res)[1])
print('Jarque-Bera p = ', scipy.stats.jarque_bera(Res)[1])
print('stderr = ', numpy.std(Res))
print('correlation = ', scipy.stats.pearsonr(normResidPrice, normResidTotal))
