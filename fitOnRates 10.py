# Fit on rates for medians of array data
# September 2017 WRB

# Import modules
import scipy
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys
import os
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
import seaborn as sns
import argparse
from joblib import Parallel, delayed
import itertools
import random
import math


########################################################
################ Parse input parameters ################
# Set up command line parser
parser = argparse.ArgumentParser(description="This script fits median fluorescence data to determine kinetic parameters.")
parser.add_argument('-CPseries', help='Filename of the CPseries file containing the cluster and fluorescence information for the experiment.',required=True)
parser.add_argument('-CPannot', help='Filename of the CPannot file for the experiment.',required=True)
parser.add_argument('-numTimepoints', help='Number of timepoints collected for the experiment', required = True)
parser.add_argument('-vFile', help='Txt file containing list of variants that you want fit.')
parser.add_argument('-nBootstrap', default=1000, help='Number of bootstraps')
parser.add_argument('-nCores', default=1, help='Number of cores to use in parallelization of fitting')
parser.add_argument('-times', help='Time Dict File Name')
parser.add_argument('-saveName', help='Prefix of the file to save the data')
parser.add_argument('-timeCutoff', help='Dont fit points after time cutoff')
parser.add_argument('-conc', default=1, help='Concentration of the experiment in nM. Input 1 if you want kobs values')
parser.add_argument('-normalize', action='store_true', default=False, help='Change if you want to normalize by the fiducial mark.')
parser.add_argument('-plot', action='store_true', default=False, help='Change if you dont want to plot')

# Sample command: python fitOnRates.py -CPseries baseline_and_association_500pM_miR_21_Ago.CPseries.pkl -CPannot BDLCD_ALL.CPannot.pkl -numTimepoints 17 -vFile miR21_Ago_variant_tableTest.txt -nCores 2 -times rates.timeDict.p -saveName /Users/winstonbecker/Google\ Drive/Research/Ago/OnRateFitting/Plots/

########################################################
########################################################
# Parse command line arguments
inputArgs = parser.parse_args()


# Set the initial parameters
nCores = int(inputArgs.nCores)
nBootstraps = int(inputArgs.nBootstrap)
figSaveLocPrefix = str(inputArgs.saveName)
numTimePoints = int(inputArgs.numTimepoints)
concentrationValue = int(inputArgs.conc)
timeCutoff = float(int(inputArgs.timeCutoff))


# Load Times and apply time cutoff
timeDictFileName = str(inputArgs.times)
timeDictfile = open(timeDictFileName, 'rb')
timeDict = pd.read_pickle(timeDictfile)
Times = timeDict['009']
Times.insert(0,0)
newTimes = []
for i in range(len(Times)):
    if Times[i]<timeCutoff:
        newTimes.append(Times[i])
Times = np.array(newTimes)


# Load Variant Table
VariantTableFileName = str(inputArgs.vFile)
VariantTable = pd.DataFrame.from_csv(VariantTableFileName, sep='\t', header=None, index_col=False)
if len(VariantTable.columns) == 4:
    VariantTable.columns = ('Sequence', 'variant_ID', 'mutant_group', 'mut_annotation')#, "a", 'b')
else:
    VariantTable.columns = ('Sequence', 'variant_ID', 'mutant_group', 'mut_annotation', "WT_sequence", 'Group_num')


# Load CPseries and set column names
CPseriesFileName = str(inputArgs.CPseries)
pkl_file = open(CPseriesFileName, 'rb')
CPseries = pd.read_pickle(pkl_file)
CPseries.index.name = 'clusterID'
colNames = []
columsToDrop = []
for i in range(numTimePoints):
    colNames.append('t' + str(i))
    if i>=len(Times):
        columsToDrop.append('t' + str(i))
CPseries.columns = colNames
CPseries = CPseries.drop(columsToDrop, axis=1)
numTimePoints = numTimePoints-len(columsToDrop)


# Load Red CPseries--Include this if you want to normalize by RNA signal
'''RedCPseriesFileName = str(inputArgs.RedCPseries)
pkl_file = open(RedCPseriesFileName, 'rb')
RedCPseries = pd.read_pickle(pkl_file)
RedCPseries.index.name = 'clusterID' '''


# Define concentration
concentrations = np.array([concentrationValue]*numTimePoints)


# Load CP annot
CPAnnotFileName = str(inputArgs.CPannot)
pkl_file = open(CPAnnotFileName, 'rb')
CPannot = pd.read_pickle(pkl_file)
CPannot.index.name = 'clusterID'


# Merge CPseries and CPannot
BindingSeries = CPseries.join(CPannot, how = 'inner')
#RedSeries = RedCPseries.join(CPannot, how = 'inner')


# Set normalization
normalize = inputArgs.normalize
if normalize:
    Fiducial = BindingSeries.groupby('variant_ID').get_group('11111111').iloc[:, 0:].median()
    normalization = Fiducial[0:numTimePoints]
else:
    normalization = [1]*numTimePoints


# Create Plots subdirectory
Plot = inputArgs.plot
plotLocation = figSaveLocPrefix + 'Plots/'
if Plot:
    plotLocation = figSaveLocPrefix + 'Plots/'
    if not os.path.isdir(plotLocation):
        print "Making new directory: " + plotLocation
        os.makedirs(plotLocation)


# Normalize points by red signal--Include this if you want to normalize by RNA signal
'''first_percentile = np.percentile(RedSeries[~RedSeries[0].isnull()][0], 1)

for i in range(len(RedSeries)):
    if RedSeries.iloc[i,0]<first_percentile:
        RedSeries.iloc[i,0] = first_percentile

for i in range(numTimePoints):
    colName = 't' + str(i)
    BindingSeries.loc[:,colName] = BindingSeries[colName]/RedSeries[0]'''


########################################################
########################################################
# Define Helper Fitting Functions

def objectiveFunctionOnRates(params, times, conc, data=None, weights=None):
    # Inputs
    # params--parameters specifying the equation, in this case kon is the only parameter
    # data--fluorescence values if specified
    # times--times corresponding to fluorescence values
    # conc--concentration corresponding to each data point
    # fmax--fmax corresponding to each data point (this should be the fluorescence value if the cluster was fully saturated)
    # fmin--fmin corresponding to each data point (this should be the fluorescence value for when nothing is bound to a cluster)
    # Outputs
    # This function will return the fit value, the residuals, or weighted residuals of on rate objective function.
    
    parvals = params.valuesdict()
    kon = parvals['kon']
    fmax = parvals['fmax']
    fmin = parvals['fmin']
    
    fluorescence = fmin + (fmax-fmin)*(1 - np.exp(-kon*conc*times))

    if data is None:
        return fluorescence
    elif weights is None:
        return fluorescence - data
    else:
        return (fluorescence - data)*weights


def initializeObjectiveFunctionOnRatesParameters(fluorescence, args):
    # Inputs
    # fluorescence--fluorescence values
    # args contains x--times corresponding to fluorescence values and conc--concentration corresponding to each data point
    # Outputs
    # params--Parameters class for lmfit containing the parameters to be fit
    # paramNames--Names of the parameters to be fit
    # finalParams--Data Structure to store the output data
    
    x = args[0]
    conc = args[1]
    
    # Define time range
    timeRange = max(x)-min(x)

    # Define minimum value for kon corresponding to a 1% increase in fluorescence over the life of the experiment.
    minkon = -np.log(0.99)/(timeRange*max(conc))

    # Define maximum value for kon corresponding to a 99% increase in fluorescence in 1/10 of the first time interval.
    firstDelta_t = x[1]-x[0]
    maxkon = -np.log(0.01)/(0.1*firstDelta_t*min(conc))
    
    # Define initial value for kon corresponding to a 50% increase in fluorescence in the first time interval.
    initkon = -np.log(0.5)/(firstDelta_t*min(conc))

    # Define initial fit parameters
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['kon', 'fmin', 'fmax'])
    
    # Set the max, min, and initial values for the fit parameters.
    fitParameters.loc[:, 'kon'] = [minkon, initkon, maxkon]

    # Define the min, max, and initial conditions for fmin and fmax
    if min(fluorescence)==0:
        fitParameters.loc[:, 'fmin'] = [0, 0, 0.2]
    else:
        fitParameters.loc[:, 'fmin'] = [min(fluorescence)*0.5, min(fluorescence), min(fluorescence)*1.4]
    
    if max(fluorescence)==0:
        fitParameters.loc[:, 'fmax'] = [0, 0, 1]
    else:
        fitParameters.loc[:, 'fmax'] = [max(fluorescence)*0.5, max(fluorescence), max(fluorescence)*1.4]
    
    # Define the names of the fit parameters.
    paramNames = fitParameters.columns.tolist()

    # Store fit parameters in Parameters class for fitting with lmfit.
    params = Parameters()
    for param in paramNames:
        params.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param])
        
    return params, paramNames


def fitKon(fluorSeries, time, concentrations, mutantID, figSaveLocPrefix, mutantDescription, plot = 1):
     if np.isnan(fluorSeries).any() == False:

        fmin = np.array([min(fluorSeries)]*len(fluorSeries))
        fmax = np.array([max(fluorSeries)]*len(fluorSeries))
        args = (time, concentrations, fmax, fmin)
        x = time
                
        params, param_names = initializeObjectiveFunctionOnRatesParameters(fluorSeries, args)
                
        func = objectiveFunctionOnRates

        results = minimize(func, params,
                args = (x, concentrations ),
                kws={'data':fluorSeries},
                xtol=1E-6, ftol=1E-6, maxfev=10000)
        
        final_params = [];
                        
        for param in param_names:
            final_params.append(results.params[param].value)
            final_params.append(results.params[param].stderr)
        
        # Compute error of fit
        fluorSeries = np.array(fluorSeries)
        ss_total = np.sum((fluorSeries - fluorSeries.mean())**2)
        ss_error = np.sum((results.residual)**2)
        rsquared = 1-ss_error/ss_total
        rmse = np.sqrt(ss_error)
        final_params.append(rsquared)
        final_params.append(results.ier)
        final_params.append(rmse)        
        
        # If you want to plot the non bootstrapped results uncomment this
        #if plot == 1:
        #    plotNonBootstrappedResults(x, concentrations, results, final_params, time, fluorSeries, mutantDescription, mutantID)

        return final_params


def plotNonBootstrappedResults(x, concentrations, results, final_params, time, fluorSeries, mutantDescription, mutantID):
    fitDataTimes = np.logspace(np.log10(min(1, x[1])), np.log10(max(x)*1.2), 128)
    fitDataConcentrations = np.array([max(concentrations)]*128)
    fitFluor = objectiveFunctionOnRates(results.params, fitDataTimes, fitDataConcentrations)
    
    sns.set_style("ticks")
    figA = plt.figure(figsize = (6*1.3, 4.5*1.3))
    plt.xlabel('Time [s]', fontsize = 18)
    plt.ylabel('Integrated Fluorescence', fontsize = 18)
    plt.plot(time, fluorSeries, 'ro')
    fit, = plt.plot(fitDataTimes, fitFluor, 'k-', label = 'Kon = '+str(round(final_params[0],2)) + '$M^{-1}$'+'$s^{-1}$')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    plt.legend(handles=[fit], loc = 'upper left', fontsize = 12)
    
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.xlim(x[0]/2, max(x)*1.3)
    
    
    plt.title(mutantDescription, fontsize = 12)
    figA.savefig(figSaveLocPrefix+mutantID+ '_' + mutantDescription +'plot.pdf', format='pdf', dpi=100)
    plt.close()

def plotBootstrappedResults(medians, medCIlow, medCIhigh, concentrations, times, allresults, figSaveLocPrefix, mutantID, mutantDescription):
    fitDataTimes = np.logspace(np.log10(min(1, times[1])), np.log10(max(times)*1.2), 80)
    fitDataConcentrations = np.array([max(concentrations)]*80)

    fitFluor = allresults[2] + (allresults[4]-allresults[2])*(1 - np.exp(-allresults[0]*fitDataConcentrations*fitDataTimes))
    fitFluor2p5 = allresults[13] + (allresults[16]-allresults[13])*(1 - np.exp(-allresults[10]*fitDataConcentrations*fitDataTimes))
    fitFluor97p5 = allresults[14] + (allresults[17]-allresults[14])*(1 - np.exp(-allresults[11]*fitDataConcentrations*fitDataTimes))
    
    sns.set_style("ticks")
    figA = plt.figure(figsize = (6*1.3, 4.5*1.3))
    plt.xlabel('Time (s)', fontsize = 18)
    plt.ylabel('Normalized Fluorescence (a.u.)', fontsize = 18)


    errorbarvalues = np.array([list(np.subtract(medians,medCIlow)), list(np.subtract(medCIhigh,medians))])
    plt.errorbar(times, medians, errorbarvalues, fmt='o', color = 'k', ecolor='k', capthick=2)

    fit, = plt.plot(fitDataTimes, fitFluor, 'k-', label = 'kobs = '+str(round(allresults[0],5)) + '$M^{-1}$'+'$s^{-1}$')
    plt.fill_between(fitDataTimes, fitFluor2p5, fitFluor97p5, alpha=0.2, facecolor = 'darkgray')


    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    plt.legend(handles=[fit], loc = 'upper left', fontsize = 12)
    
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.xlim(times[0]/2, max(times)*1.3)
        
    plt.title(mutantDescription, fontsize = 12)
    figA.savefig(figSaveLocPrefix+mutantID+ '_' + mutantDescription +'_plot.pdf', format='pdf', dpi=100)
    plt.close()


def bootstrapMedianCI(variantSeries, times, numBootstraps = 1000):
    numTimes = len(times)
    medians = []
    medCIlow = []
    medCIhigh = []
    
    for j in xrange(numTimes):
        bootstrapMedian = []
        currentTime = variantSeries.iloc[:, j]
        NumClusters = len(currentTime)

        '''# remove outliers using median standard deviation
        # http://www.sciencedirect.com/science/article/pii/S0022103113000668
        b = 1.4826 # Include to assume normality of data
        # b = 1
        median = np.nanmedian(currentTime)
        med_abs_deviation = b*np.nanmedian(np.abs(currentTime - np.nanmedian(currentTime)))
        scores = np.abs(currentTime - median) / med_abs_deviation
        notOutliers = np.array(scores<6)

        currentTime = currentTime[notOutliers]

        # remove outliers with unreasonable values
        currentTime = currentTime[currentTime<10]'''

        numSamples = len(currentTime)
        for i in xrange(numBootstraps):
            bootstrapMedian.append(np.nanmedian(np.random.choice(currentTime, numSamples)))
        
        medians.append(np.nanmedian(currentTime))
        medCIlow.append(np.percentile(bootstrapMedian, 2.5))
        medCIhigh.append(np.percentile(bootstrapMedian, 97.5))
    return medians, medCIlow, medCIhigh, NumClusters

def fitkonWrapperBootstrap(mutantID, BindingSeries, times, concentrations, figSaveLocPrefix, mutantDescription, normalization, numBootstraps = 1000, plotBootstrap = True):
    # Check to see if there are any binding curves matching the variant ID, else return a row of zeros
    if any(BindingSeries['variant_ID'].str.contains(mutantID)):
        # Define The VariantSeries by extracting all binding curves matcing the variant ID
        # variantSeries = BindingSeries.groupby('variant_ID').get_group(mutantID).iloc[:, 0:] # 11/13 edited to find multiple variant IDs at once
        variantSeries = BindingSeries[BindingSeries['variant_ID'].str.contains(mutantID)].iloc[:, 0:]

        # Probably would be better to just do this for the entire data frame
        # Check for all nan values in the binding curves matcing the variant ID and return a row of zeros if all rows contain only nan values, else remove all of the nan rows
        if variantSeries.isnull().all().all():
            return [mutantID]+ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            #variantSeries = variantSeries.loc[~variantSeries.isnull().all(axis=1)]
            variantSeries = variantSeries[~variantSeries.T.isnull().any()]
            
        # Check for all 0 values in the binding curves matcing the variant ID and return a row of zeros if all rows contain only 0 values, else remove all of the 0 rows
        if (variantSeries==0).all().all():
            return [mutantID]+ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            variantSeries = variantSeries.loc[(variantSeries!=0).any(axis=1)]
            #variantSeries = variantSeries[(variantSeries.T != 0).any()]

        # Only run if there are as many time points in the normalization as in the binding curves of interest, else return a row of zeros
        if len(np.array(variantSeries.median()))==len(np.array(normalization)):
            # Find the bootstraped median values for each point
            medians, medCIlow, medCIhigh, NumClusters = bootstrapMedianCI(variantSeries, times, numBootstraps)
            numTimepoints = len(Times)        

            # Bootstrap the fits
            bootstrappedMedians = [] # Initialize list to store bootstrapped values
            # Boostrap numBootstraps times
            for j in xrange(numBootstraps):
                # During each instance of bootstrapping, sample the data with replacement
                mediansValsThisIteration = np.array(variantSeries.sample(frac = 1.0, replace = True).median())/np.array(normalization)

                # Fit the normalized, bootstrapped median values
                results = fitKon(mediansValsThisIteration, times, concentrations, mutantID, figSaveLocPrefix, mutantDescription, plot = 0)

                # Append only if you get a result--this is probably redundent now that all nan rows are removed at the beginning, but I left it in there anyway
                if results is not None:
                    bootstrappedMedians.append(results)
            
            # If a none object results from the bootstrapping, return zeros
            if bootstrappedMedians is None:
                return [mutantID]+ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            # Convert list of lists to a data frame
            bootstrappedMedians = pd.DataFrame(bootstrappedMedians)

            # If a non object results from conversion to a data frame, or if it is filled with nan values, return zeros--again this is probably redundent error checking
            if bootstrappedMedians is None or bootstrappedMedians.isnull().values.any():
                return [mutantID]+ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            # Define the confidence intervals from the bootstraped fits
            bootstrappedConfidenceIntervalsKon = np.percentile(bootstrappedMedians[0], [50, 2.5, 97.5])
            bootstrappedConfidenceIntervalsfmin = np.percentile(bootstrappedMedians[2], [50, 2.5, 97.5])
            bootstrappedConfidenceIntervalsfmax = np.percentile(bootstrappedMedians[4], [50, 2.5, 97.5])

            # Compute the medians and fit the medians
            mediansVals = np.array(variantSeries.median())/np.array(normalization)
            allresults = fitKon(mediansVals, times, concentrations, mutantID, figSaveLocPrefix, mutantDescription, plot = 0)
            
            # Combine the results of the median fit and the bootstrapped fits
            allresults = allresults+list(bootstrappedConfidenceIntervalsKon)+list(bootstrappedConfidenceIntervalsfmin) + list(bootstrappedConfidenceIntervalsfmax)
            allresults.append(NumClusters)

            # Add bootstrapped fmax values
            all_bootstrapped_fmax_values = list(bootstrappedMedians[4])
            allresults.append(list(all_bootstrapped_fmax_values))

            # Return a row of zeros if the results are empty--again redundant error checking
            if allresults == []:
                allresults = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif plotBootstrap:
                plotBootstrappedResults(medians/np.array(normalization), medCIlow/np.array(normalization), medCIhigh/np.array(normalization), concentrations, times, allresults, figSaveLocPrefix, mutantID, mutantDescription)

            # Add the mutant ID to the list of results and return the results
            allresults = [mutantID]+allresults
            return allresults
        else:
            return [mutantID]+ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        return [mutantID]+ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


##################################################################
##################################################################
# Main
# Fit the selected variables to the clusters
if nCores > 1:
    # Fit the parameters to the user defined functions to the fluorescence values for single clusters in parallel.
    results = Parallel(n_jobs=nCores, verbose = 9)(delayed(fitkonWrapperBootstrap)(VariantTable['variant_ID'][i], BindingSeries, np.array(Times), concentrations, plotLocation, VariantTable[VariantTable['variant_ID'] == VariantTable['variant_ID'][i]].iloc[0,3], normalization, numBootstraps = nBootstraps, plotBootstrap = Plot) for i in range(len(VariantTable)))
    results = pd.DataFrame(results)
    results.columns = ('variant_ID', 'kobs', 'kobs_err', 'fmin', 'fminerror', 'fmax', 'fmaxerror','rsquared', 'ier', 'rmse', 'Kon_50', 'Kon_2p5', 'Kon_97p5', 'fmin_50', 'fmin_2p5', 'fmin_97p5', 'fmax_50', 'fmax_2p5', 'fmax_97p5', 'nClusters', 'fmax_values')
    results = pd.merge(VariantTable, results, on = 'variant_ID')
    results.to_csv(figSaveLocPrefix+"fitOnRates.csv", index =False)
else:
    # Non-parallelized version:
    VariantTable[['variant_ID', 'kobs', 'kobs_err', 'fmin', 'fminerror', 'fmax', 'fmaxerror','rsquared', 'ier', 'rmse', 'Kon_50', 'Kon_2p5', 'Kon_97p5', 'fmin_50', 'fmin_2p5', 'fmin_97p5', 'fmax_50', 'fmax_2p5', 'fmax_97p5', 'nClusters', 'fmax_values']] = VariantTable['variant_ID'].apply(lambda x: fitkonWrapperBootstrap(x, BindingSeries, np.array(Times), concentrations, plotLocation, VariantTable[VariantTable['variant_ID'] == x].iloc[0,3], normalization, numBootstraps = nBootstraps, plotBootstrap = Plot)).apply(pd.Series)
    VariantTable.to_csv(figSaveLocPrefix+"fitOnRates.csv", index =False)

