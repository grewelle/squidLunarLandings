import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
import csv
from scipy.optimize import minimize
from scipy import stats


SMALL_SIZE = 24
MEDIUM_SIZE = 32
BIGGER_SIZE = 48

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

en = input("Which regression is preferred?  Normal (A), Robust (B), Cauchy (C):  ")


"""Function hill imports a starting value for 3 parameters and
returns the sum of the residuals squared to be minimized"""
def hill(x, time, z, w, mu):

    c = 2*np.pi/29.5

    if en == 'B' or en == 'b':
        reg = sum(w[i]*2*((1+(x[0]*c*np.sin(c*(x[1]+(i+29.5)))/(c**2+mu**2)+mu*x[0]*np.cos(c*(x[1]+(i+29.5)))/(c**2+mu**2)+x[2]*np.exp(-mu*(i+29.5)) - z[i])**2)**.5-1) for i in range(len(time)))
    elif en == 'C' or en == 'c':
        reg = sum(w[i]*np.log(1+(x[0]*c*np.sin(c*(x[1]+(i+29.5)))/(c**2+mu**2)+mu*x[0]*np.cos(c*(x[1]+(i+29.5)))/(c**2+mu**2)+x[2]*np.exp(-mu*(i+29.5)) - z[i])**2) for i in range(len(time)))
    else:
        reg = sum(w[i]*(x[0]*c*np.sin(c*(x[1]+(i+29.5)))/(c**2+mu**2)+mu*x[0]*np.cos(c*(x[1]+(i+29.5)))/(c**2+mu**2)+x[2]*np.exp(-mu*(i+29.5)) - z[i])**2 for i in range(len(time)))

    return reg

def jackknife(fits, pseudoFits):
    residA = []
    residB = []
    residC = []
    sumA = 0
    sumB = 0
    sumC = 0
    tot = len(pseudoFits)
    denom = tot * (tot - 1)
    for s in range(len(pseudoFits)):
        residA.append(tot * fits[0] - (tot - 1) * pseudoFits[s][0])
        residB.append(tot * fits[1] - (tot - 1) * pseudoFits[s][1])
        residC.append(tot * fits[2] - (tot - 1) * pseudoFits[s][2])
    for h in range(len(residA)):
        sumA += (residA[h] - np.average(residA)) ** 2
        sumB += (residB[h] - np.average(residB)) ** 2
        sumC += (residC[h] - np.average(residC)) ** 2
    return [np.sqrt(sumA / denom), np.sqrt(sumB / denom), np.sqrt(sumC / denom)]

def main():

    abs_file_path = "C:/Users/Richard/Desktop/desktop/Squids/FinalSquidCalculations_BB.csv"
    with open(abs_file_path, newline='') as csvfile:
        totalData = list(csv.reader(csvfile))


    lunarIndex = list(np.array(totalData)[1:,1])
    cpue = list(np.array(totalData)[1:,23])
    raw_weights = list(np.array(totalData)[1:,11])
    mu=0.2




    for i in range(len(lunarIndex)):
        lunarIndex[i] = float(lunarIndex[i])
        cpue[i] = float(cpue[i])
        raw_weights[i] = float(raw_weights[i])




    norm_weights = []
    for x in range(len(raw_weights)):
        norm_weights.append(raw_weights[x]/np.sum(raw_weights))

    siz=list(5*np.array(raw_weights))

    harmonics = []
    amplitude = []
    period = []

    for s in range(60):

        x0 = np.array([0.5, 2.0, 0.05])  # initial guess for curve fit parameters
        # weighted least squares nonlinear regression with bounds
        res = minimize(hill, x0, args=(lunarIndex, cpue, norm_weights, mu), method='SLSQP',
                       bounds=((0, None), (0, None), (None, None)), tol=1e-6)
        params = res.x  # give parameters a variable name
        print(params)



        pseudoFits = []
        for r in range(len(lunarIndex)):
            pseudoWeight = norm_weights[0:r] + norm_weights[r + 1:]
            pseudoLunarIndex = lunarIndex[0:r] + lunarIndex[r + 1:]
            pseudoCPUE = cpue[0:r] + cpue[r + 1:]

            pseudoRes = minimize(hill, x0, args=(pseudoLunarIndex, pseudoCPUE, pseudoWeight, mu), method='SLSQP',
                                 bounds=((0, None), (0, None), (None, None)), tol=1e-6)
            pseudoFits.append(pseudoRes.x)

        paramError = jackknife(params, pseudoFits)
        print(paramError)

        tt = (params[0] -0) / paramError[0]  # t-statistic for mean
        pval = stats.t.sf(np.abs(tt), len(lunarIndex) - 1)   # two-sided pvalue = Prob(abs(t)>tt)
        print('amplitude t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))

        tt2 = (params[1] - 14.5) / paramError[1]  # t-statistic for mean
        pval2 = stats.t.sf(np.abs(tt2), len(lunarIndex) - 1)  # two-sided pvalue = Prob(abs(t)>tt)
        print('period t-statistic = %6.3f pvalue = %6.4f' % (tt2, pval2))

        harmonics.append(stats.hmean([pval,pval2]))
        amplitude.append(pval)
        period.append(pval2)
        mu += .1

    fig2 = plt.figure(figsize=(10, 10))
    index = np.linspace(0.2, 6.2, 60)
    plt.plot(index, amplitude, linewidth='3')

    plt.xlabel('Death Rate at Spawning (day-1)')
    plt.ylabel('Amplitude P-value')
    plt.show()
    fig2.savefig("amplitude.tiff")


main()
