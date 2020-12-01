import pandas
import numpy
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot

def convert(x):
    return numpy.log(1 + x/100)

# Preliminary Analysis
PriceDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'PriceReturns')
TotalDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'TotalReturns')
SizeDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'Size')
BillDF = pandas.read_excel('SizeDeciles.xlsx', sheet_name = 'Bill')
priceRet = PriceDF.values[:, 1:]
totalRet = TotalDF.values[:, 1:]
size = SizeDF.values[:, 1:]
bill = BillDF.values[:, 1]
RiskFree = convert(bill/12)
lsize = numpy.log(size)
N = 10
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
pyplot.title('beta vs weights')
pyplot.show()
pyplot.plot(numpy.log(weightList), numpy.log(abs(empiricalBetaPrice)), 'go')
pyplot.title('log beta vs log weights')
pyplot.show()
RegLogBeta = scipy.stats.linregress(numpy.log(weightList), numpy.log(abs(empiricalBetaPrice)))
print(RegLogBeta)
priceA = RegLogBeta.intercept
priceB = RegLogBeta.slope
print('intercept = ', priceA)
print('slope = ', priceB)
pyplot.plot(weightList, empiricalBetaPrice / weightList ** priceB, 'go')
pyplot.title('normalized beta vs weights')
pyplot.show()
priceS = numpy.std(empiricalBetaPrice / weightList)
print('stdev = ', priceS)

# Empirical Beta vs Weights Analysis, Mean Trend
pyplot.plot(weightList, empiricalBetaTotal, 'go')
pyplot.title('beta vs weights')
pyplot.show()
pyplot.plot(numpy.log(weightList), numpy.log(abs(empiricalBetaTotal)), 'go')
pyplot.title('log beta vs log weights')
pyplot.show()
RegLogBeta = scipy.stats.linregress(numpy.log(weightList), numpy.log(abs(empiricalBetaTotal)))
print(RegLogBeta)
totalA = RegLogBeta.intercept
totalB = RegLogBeta.slope
print('intercept = ', totalA)
print('slope = ', totalB)
pyplot.plot(weightList, empiricalBetaTotal / weightList ** totalB, 'go')
pyplot.title('normalized beta vs weights')
pyplot.show()
totalS = numpy.std(empiricalBetaTotal / weightList)
print('stdev = ', totalS)

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

# Power Scaling Empirical Beta
print('analysis for price returns')
NewBenchmark = numpy.array([(weightList[item] ** priceB) * PriceBenchmark[item] for item in range(T*(N-1))])
residAlphaPrice = PriceReturns - numpy.exp(priceA) * NewBenchmark - PriceBenchmark
pyplot.plot(weightList, residAlphaPrice, 'go')
pyplot.title('residuals vs weights')
pyplot.show()
pyplot.plot(weightList, numpy.log(abs(residAlphaPrice)), 'go')
pyplot.title('log residuals vs weights')
pyplot.show()
RegLogAlpha = scipy.stats.linregress(weightList, numpy.log(abs(residAlphaPrice)))
gammaPrice = RegLogAlpha.slope
print('power = ', gammaPrice)
normResidPrice = residAlphaPrice * numpy.exp(- gammaPrice * weightList)
pyplot.plot(weightList, normResidPrice, 'go')
pyplot.title('normalized alpha vs weights')
pyplot.show()
qqplot(normResidPrice, line = 's')
pyplot.show()
plot_acf(normResidPrice)
pyplot.show()
print('Shapiro-Wilk p = ', scipy.stats.shapiro(normResidPrice)[1])
print('Jarque-Bera p = ', scipy.stats.jarque_bera(normResidPrice)[1])
sigmaPrice = numpy.std(normResidPrice)
print('sigma = ', sigmaPrice)

# Power Scaling Empirical Beta
print('analysis for total returns')
NewBenchmark = numpy.array([(weightList[item] ** totalB) * TotalBenchmark[item] for item in range(T*(N-1))])
residAlpha = TotalReturns - numpy.exp(totalA) * NewBenchmark - TotalBenchmark
pyplot.plot(weightList, residAlpha, 'go')
pyplot.title('residuals vs weights')
pyplot.show()
pyplot.plot(weightList, numpy.log(abs(residAlpha)), 'go')
pyplot.title('log residuals vs weights')
pyplot.show()
RegLogAlpha = scipy.stats.linregress(weightList, numpy.log(abs(residAlpha)))
gammaTotal = RegLogAlpha.slope
print('power = ', gammaTotal)
normResidTotal = residAlpha * numpy.exp(- gammaTotal * weightList)
pyplot.plot(weightList, normResidTotal, 'go')
pyplot.title('normalized alpha vs weights')
pyplot.show()
qqplot(normResidTotal, line = 's')
pyplot.show()
plot_acf(normResidTotal)
pyplot.show()
print('Shapiro-Wilk p = ', scipy.stats.shapiro(normResidTotal)[1])
print('Jarque-Bera p = ', scipy.stats.jarque_bera(normResidTotal)[1])
sigmaTotal = numpy.std(normResidTotal)
print('sigma = ', sigmaTotal)

print('correlation = ', scipy.stats.pearsonr(normResidPrice, normResidTotal))

# # Linear Scaling Empirical Beta
# NewBenchmark = numpy.array([weightList[item] * Benchmark[item] for item in range(T*(N-1))])
# residAlpha = Returns - S * NewBenchmark - Benchmark
# pyplot.plot(weightList, residAlpha, 'go')
# pyplot.title('residuals vs weights')
# pyplot.show()
# pyplot.plot(weightList, numpy.log(abs(residAlpha)), 'go')
# pyplot.title('log residuals vs weights')
# pyplot.show()
# RegLogAlpha = scipy.stats.linregress(weightList, numpy.log(abs(residAlpha)))
# gamma = RegLogAlpha.slope
# print('power = ', gamma)
# normResid = residAlpha * numpy.exp(- gamma * weightList)
# pyplot.plot(weightList, normResid, 'go')
# pyplot.title('normalized alpha vs weights')
# pyplot.show()
# qqplot(normResid, line = 's')
# pyplot.show()
# plot_acf(normResid)
# pyplot.show()
# print('Shapiro-Wilk p = ', scipy.stats.shapiro(normResid)[1])
# print('Jarque-Bera p = ', scipy.stats.jarque_bera(normResid)[1])
# sigma = numpy.std(normResid)
# print('sigma = ', sigma)