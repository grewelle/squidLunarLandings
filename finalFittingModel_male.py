import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns; sns.set(style="white", color_codes=True)
import csv
from scipy.optimize import minimize


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


"""Function hill imports a starting value for 3 parameters (thetaMax, hill coefficient, and EC50) and
returns the sum of the residuals squared to be minimized"""
def hill(x, time, z, w):

    c = 2*np.pi/29.5

    if en == 'B' or en == 'b':
        reg = sum(w[i]*2*((1+(x[0]*c*np.sin(c*(x[1]+(i+29.5)))/(c**2+.36)+.6*x[0]*np.cos(c*(x[1]+(i+29.5)))/(c**2+.36)+x[2]*np.exp(-.6*(i+29.5)) - z[i])**2)**.5-1) for i in range(len(time)))
    elif en == 'C' or en == 'c':
        reg = sum(w[i]*np.log(1+(x[0]*c*np.sin(c*(x[1]+(i+29.5)))/(c**2+.36)+.6*x[0]*np.cos(c*(x[1]+(i+29.5)))/(c**2+.36)+x[2]*np.exp(-.6*(i+29.5)) - z[i])**2) for i in range(len(time)))
    else:
        reg = sum(w[i]*(x[0]*c*np.sin(c*(x[1]+(i+29.5)))/(c**2+.36)+.6*x[0]*np.cos(c*(x[1]+(i+29.5)))/(c**2+.36)+x[2]*np.exp(-.6*(i+29.5)) - z[i])**2 for i in range(len(time)))

    return reg

"""Function plotIndFits imports parameters, concentration and response values, and a binary value for whether
to plot on a semilog plot and plots the fitted curve with the data"""
def plotIndFits(params, lunarDays, cpue, sizes, error):
    fig2 = plt.figure(figsize=(10, 10))
    index = np.linspace(0, 29.5, round(29.5 * 10))
    c = 2 * np.pi / 29.5
    cov = 0
    func = params[0] * np.cos((index + params[1]) * 2 * np.pi / 29.5)
    func2 = params[0] * c * np.sin(c * (params[1] + index)) / (c ** 2 + .36) + .6 * params[0] * np.cos(
        c * (params[1] + index)) / (c ** 2 + .36) + params[2] * np.exp(-.6 * index)
    func3 = params[0] * c * np.sin(c * (params[1] + (index + 295))) / (c ** 2 + .36) + .6 * params[0] * np.cos(
        c * (params[1] + (index + 295))) / (c ** 2 + .36) + params[2] * np.exp(-.6 * (index + 295))
    print((29.5 * 10 - (np.argmax(func3) + 1)) / 10,
          ((1 + func3[np.argmax(func3)]) - (1 - func3[np.argmax(func3)])) / (1 - func3[np.argmax(func3)]))
    d0 = c * np.sin(c * (params[1] + index)) / (c ** 2 + .36) + .6  * np.cos(
        c * (params[1] + index)) / (c ** 2 + .36)
    d1 = params[0] * c * np.cos(c * (params[1] + index)) / (c ** 2 + .36) - .6 * params[0] * np.sin(
        c * (params[1] + index)) / (c ** 2 + .36)
    d2 = np.exp(-.6*index)
    errFunc = np.sqrt(error[0]**2*d0**2 + (error[1]/1)**2*d1**2 + error[2]**2*d2**2 + 2*cov*d0*d1 + 1.2)
    errFuncTop = func2 + errFunc
    errFuncBottom = func2 - errFunc
    plt.plot(index, func2, label='CPUE best fit model', linewidth = '3')
    plt.plot(index, func, '--', color = 'black', label='Influx rate of spawning squid', linewidth = '3')
    #plt.plot(index,errFuncTop)
    #plt.plot(index,errFuncBottom)
    colorful = colors.Colormap('Greys')
    plt.fill_between(index, errFuncBottom, errFuncTop, alpha=.3)
    sns.scatterplot(lunarDays, cpue, s=sizes, palette=colorful)
    plt.xlabel('Days from Lunar Peak')
    plt.ylabel('Seasonal Adjusted CPUE')
    plt.legend()
    plt.show()
    fig2.savefig("MaleSquid.tiff")

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
    cpue = list(np.array(totalData)[1:,26])
    raw_weights = list(np.array(totalData)[1:,13])



    lunarIndex = lunarIndex[:28] + lunarIndex[29:]
    cpue = cpue[:28] + cpue[29:]
    raw_weights = raw_weights[:28] + raw_weights[29:]



    for i in range(len(lunarIndex)):
        lunarIndex[i] = float(lunarIndex[i])
        cpue[i] = float(cpue[i])
        raw_weights[i] = float(raw_weights[i])




    norm_weights = []
    for x in range(len(raw_weights)):
        norm_weights.append(raw_weights[x]/np.sum(raw_weights))

    siz=list(5*np.array(raw_weights))


    x0 = np.array([0.5, 2.0, 0.05])  # initial guess for curve fit parameters
    # weighted least squares nonlinear regression with bounds
    res = minimize(hill, x0, args=(lunarIndex, cpue, norm_weights), method='SLSQP',
                   bounds=((0, None), (0, None), (None, None)), tol=1e-6)
    params = res.x  # give parameters a variable name
    print(params)


    pseudoFits = []
    for r in range(len(lunarIndex)):
        pseudoWeight = norm_weights[0:r] + norm_weights[r + 1:]
        pseudoLunarIndex = lunarIndex[0:r] + lunarIndex[r + 1:]
        pseudoCPUE = cpue[0:r] + cpue[r + 1:]

        pseudoRes = minimize(hill, x0, args=(pseudoLunarIndex, pseudoCPUE, pseudoWeight), method='SLSQP',
                             bounds=((0, None), (0, None), (None, None)), tol=1e-6)
        pseudoFits.append(pseudoRes.x)

    paramError = jackknife(params, pseudoFits)
    print(paramError)
    plotIndFits(params, lunarIndex, cpue, siz, paramError)  # plot curves with data



main()