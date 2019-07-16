# July 2017
# Winston Becker
# This script take in counts for timepoints from RISC-CNS experiments and normalizes it and fits single exponentials to the. 

#############################################
#############################################
# Import Packages

import scipy
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import time
import argparse
from string import maketrans
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
sns.set_style('ticks')

#############################################
#############################################
# Function Definitions

def objectiveFunctionCleaveRates(params, times, data=None, weights=None):
    # Objective function of single exponential cleavage rates
    # Inputs
    # params--parameters specifying the equation, in this case kon is the only parameter
    # data--fluorescence values if specified
    # times--times corresponding to fluorescence values
    # Outputs
    # This function will return the fit value, the residuals, or weighted residuals of on rate objective function.
    
    parvals = params.valuesdict()
    kc = parvals['kc']
    fmax = parvals['fmax']
    fmin = parvals['fmin']
    fluorescence = fmin + (fmax-fmin)*(np.exp(-kc*times))

    if data is None:
        return fluorescence
    elif weights is None:
        return fluorescence - data
    else:
        return (fluorescence - data)*weights


def initializeObjectiveFunctionCleaveRatesParameters(fluorescence):
    # Function to define parameters class for lmfit
    # Inputs
    # fluorescence--measured values (counts) from the experiment
    # Outputs
    # params--Parameters class for lmfit containing the parameters to be fit
    # paramNames--Names of the parameters to be fit

    # Define initial fit parameters
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['kc', 'fmin', 'fmax'])
    
    # Set the max, min, and initial values for the fit parameters.
    fitParameters.loc[:, 'kc'] = [0.00001, 0.01, 1]
    if min(fluorescence)==0:
        fitParameters.loc[:, 'fmin'] = [0, 0, 0.2]
    else:
        fitParameters.loc[:, 'fmin'] = [0, 0.1*max(fluorescence), 0.5*max(fluorescence)] #11012018
    if max(fluorescence)==0:
        fitParameters.loc[:, 'fmax'] = [0, 0, 1]
    else:
        fitParameters.loc[:, 'fmax'] = [max(fluorescence)*0.8, max(fluorescence), max(fluorescence)*1.2]
    
    # Define the names of the fit parameters.
    paramNames = fitParameters.columns.tolist()

    # Store fit parameters in Parameters class for fitting with lmfit.
    params = Parameters()
    for param in paramNames:
        params.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param])
        
    return params, paramNames




def fitKcleave(plotValues, timePoints, figurelocation, sequence, plot = 1):
    # function that fits the cleavage rate of a single variant and plots the result (pass plot=0 to turn off plotting)
    
    dontFit = 0 # initialize a variable to prevent fitting if not appropriate
    
    # Check that conditions of sufficient sequencing and signal change are met before fitting
    if np.mean(plotValues) <= 75 and plotValues[0] <= 75:
        dontFit = 1
        final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,0,0,0]
    if np.float(plotValues[0]) < np.float(5) or np.float(plotValues[1]) < np.float(5):
        dontFit = 1
        final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,0,0,0]
    if np.max(plotValues)-np.min(plotValues)<0.2*np.median(plotValues):
        final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,1,0,0]
        dontFit = 1

    # If conditions are met, fit the cleavage rate
    if np.isnan(plotValues).any() == False and dontFit == 0:
        x = timePoints
        params, param_names = initializeObjectiveFunctionCleaveRatesParameters(plotValues)
        func = objectiveFunctionCleaveRates
        results = minimize(func, params,
                args = (x, ),
                kws={'data':plotValues},method = 'differential_evolution')
        
        final_params = [];
        final_params.append(sequence)  
        for param in param_names:
            final_params.append(results.params[param].value)
            final_params.append(results.params[param].stderr)
        
        # Compute error of fit
        plotValues = np.array(plotValues)
        ss_total = np.sum((plotValues - plotValues.mean())**2)
        ss_error = np.sum((results.residual)**2)
        if ss_total>0:
            rsquared = 1-ss_error/ss_total
        else:
            rsquared = 0
        rmse = np.sqrt(ss_error)
        final_params.append(rsquared)
        final_params.append(1)
        final_params.append(rmse)  
        
        # Test conditions to identify non cleaving sequences--conditions are somewhat redundant/remants of previous iterations
        if results.params['kc'].value>0.1:
            if results.params['fmax'].value*0.5<results.params['fmin'].value:
                final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,1,0,0]
        if results.params['kc'].value>0.01:
            if results.params['fmax'].value*0.667<results.params['fmin'].value:
                final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,1,0,0]
        if rsquared<0.7:
            if np.max(plotValues)-np.min(plotValues)<50 and np.mean(plotValues)>100:
                final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,1,0,0]
            if results.params['fmax'].value*0.8<results.params['fmin'].value:
                final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,1,0,0]
        if results.params['fmax'].value*0.9<results.params['fmin'].value:
            final_params = [sequence, 0.00000001,0,0,0,np.mean(plotValues),0,1,0,0]

    if plot == 1:
        # Define fit values
        x = timePoints
        fitDataTimes = np.logspace(0.1, np.log10(max(x)+1000), 128)
        if dontFit == 0:
            fitFluor = objectiveFunctionCleaveRates(results.params, fitDataTimes)
        else:
            fitFluor = [0]*128

        # Set up figure
        figA = plt.figure(figsize=(7,5.5))
        plt.title(sequence)
        plt.gcf().subplots_adjust(bottom=0.27)
        plt.gcf().subplots_adjust(left=0.25)
        ax = plt.subplot(111)   
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        ax.xaxis.set_tick_params(length=10, width=5)
        ax.yaxis.set_tick_params(length=10, width=5)
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontname('Arial')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontname('Arial')
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)

        p0, = plt.plot(timePoints, plotValues, 'ko', label = 'Experimental', markersize=12)
        fit, = plt.plot(fitDataTimes, fitFluor, 'k-', label = 'Kc = '+str(round(final_params[1],5)) + '$s^{-1}$')
        ax.set_xlim([0,2000])
        ax.set_xticks([0,500,1000,1500,2000])
        ax.set_ylim([0,max(plotValues)*1.1])
        plt.xlabel('time (s)', fontweight='bold', fontsize = 20, fontname = 'Arial')
        plt.ylabel('Normalized Counts', fontweight='bold', fontsize = 20, fontname = 'Arial')
        plt.legend(handles=[fit], loc = 'upper left', fontsize = 12)
        figA.savefig(figurelocation+sequence+'2000.pdf', dpi=100)
        plt.close()
    else:
        if dontFit==0:
            final_params = []
    return final_params


def generateSeries(seriesList, fastqPrefix, expTableName, experimentName):
    # Add the individual timepoint counts to dataframe containing all timepoints
    for j in range(len(seriesList)):
        currentCountTable = pd.read_table(fastqPrefix + seriesList[0][j] + '_counts.tsv')
        currentCountTable=currentCountTable.rename(columns = {'counts':'t' + str(j)})
        if j == 0:
            seriesCountTable = currentCountTable
        else:
            seriesCountTable = seriesCountTable.merge(currentCountTable[list(['sequence', 't' + str(j)])], on = 'sequence')
    seriesCountTable = seriesCountTable.reset_index(drop = True)
    seriesCountTable.to_csv(expTableName + 'SeriesCounts'+experimentName+'.tsv', sep="\t", index = False)

    return seriesCountTable


def normalizeDataSeries(seriesCountTable, normalizationSequences, numTimePoints, expTableName, experimentName):
    # Function to normalize the counts for sequencing depth based on a list of normalization sequences that should not be cleaved
    controlSeq = seriesCountTable[seriesCountTable['sequence'].isin(normalizationSequences)]
    for j in range(1, numTimePoints):
        for k in range(len(seriesCountTable)):
            seriesCountTable.set_value(k, 't'+str(j), float(controlSeq['t0'].median())*float(seriesCountTable['t'+str(j)][k])/(float(controlSeq['t'+str(j)].median())+0.0001))
    for k in range(len(seriesCountTable)):
        seriesCountTable.set_value(k, 't'+str(0), float(controlSeq['t0'].median())*float(seriesCountTable['t0'][k])/(float(controlSeq['t'+str(0)].median())+0.0001))
    seriesCountTable.to_csv(expTableName + 'NormalizedSeriesCounts'+experimentName+'.tsv', sep="\t", index = False)

    return seriesCountTable


def fitData(timePoints, numTimePoints, figurelocation, experimentName, expTableName, normalizedCountTable):
    # Function to fit all of the cleavage data
    timePoints = timePoints[0].astype(float)
    timePoints = np.array(list(timePoints))
    allresults = []
    for i in range(len(normalizedCountTable)):
        plotValues = []
        for j in range(numTimePoints):
            plotValues.append(float(normalizedCountTable['t'+str(j)][i]))
        sequence = normalizedCountTable['sequence'][i]

        plotValues = list(plotValues)
        final_params = fitKcleave(plotValues, timePoints, figurelocation, sequence, plot = 1)
        if final_params == []:
            allresults.append([sequence, 0,0,0,0,0,0,0,0,0])
        else:
            allresults.append(final_params)

    allresults = pd.DataFrame(allresults)
    allresults.to_csv(expTableName + 'FitParameters'+experimentName+'.tsv', sep="\t", index = False)

    return allresults



#############################################
#############################################
# Definition of main, which includes hardcoded inputs rather than using argparse

def main(miR21, let7):
    # Read data for specified miRNA
    if mir21:
        experimentName = '8nM_37C_blocking_miR21'
        numTimePoints = 9
        timePoints = pd.read_table('/raid/USRdirs/ago/Ago/CleaveAndSeq/miR21_blocking/new_fitting_methods/single_exponential_8point_normalization_partial_test/timePoints.txt', header=None)

        expTableName = '/raid/USRdirs/ago/Ago/CleaveAndSeq/miR21_blocking/new_fitting_methods/single_exponential_8point_normalization_partial_test/'
        seriesList = pd.read_table('/raid/USRdirs/ago/Ago/CleaveAndSeq/miR21_blocking/new_fitting_methods/single_exponential_8point_normalization_partial_test/seriesList'+experimentName+'.txt', header=None)
        variantTableFilename = '/raid/USRdirs/ago/miR21_Ago_variant_table.txt'
        fastqPrefix = '/raid/USRdirs/ago/Ago/CleaveAndSeq/miR21_blocking/Counts/'
        figurelocation = '/raid/USRdirs/ago/Ago/CleaveAndSeq/miR21_blocking/new_fitting_methods/single_exponential_8point_normalization_partial_test/single_exponential_fit_figures_'+experimentName+'/'
        # 2,3,4,5,6,7,8,9-11, 6,7,8-10
        normalizationSequence = ['AAAAACAACATCAGTCACTATAGCTAAAAAA',
                               'AAAAACAACATCAGTGACTATAGCTAAAAAA',
                               'AAAAACAACATCAGTCACTAAAGCTAAAAAA',
                               'AAAAACAACATCAGTGACTAAAGCTAAAAAA',
                               'AAAAACAACATCAGTCACTTAAGCTAAAAAA',
                               'AAAAACAACATCAGTGACTTAAGCTAAAAAA',
                               'AAAAACAACATCAGTGACATAAGCTAAAAAA',
                               'AAAAACAACATCAGTGACTATAGCTAAAAAA',
                               'AAAAACAACATCAGTGACTATTGCTAAAAAA',
                               'AAAAACAACATCAGTGACTATTCCTAAAAAA',
                               'AAAAACAACATCAGTGACTATTCGTAAAAAA',
                               'AAAAACAACATCAGTGATAAGCTAAAAAA']
    elif let7:
        experimentName = '4nM_37C_blocking_let7'
        numTimePoints = 9
        timePoints = pd.read_table('/raid/USRdirs/ago/Ago/CleaveAndSeq/let7_blocking/final_analysis/final_single_exponential_partial_series_final_norm/timePoints.txt', header=None)

        expTableName = '/raid/USRdirs/ago/Ago/CleaveAndSeq/let7_blocking/final_analysis/final_single_exponential_partial_series_final_norm_final_initialization/'
        seriesList = pd.read_table('/raid/USRdirs/ago/Ago/CleaveAndSeq/let7_blocking/final_analysis/final_single_exponential_partial_series_final_norm/seriesList'+experimentName+'.txt', header=None)
        variantTableFilename = '/raid/USRdirs/ago/let7_Ago_variant_table.txt'
        fastqPrefix = '/raid/USRdirs/ago/Ago/CleaveAndSeq/let7_blocking/final_analysis/fastqs/'
        figurelocation = '/raid/USRdirs/ago/Ago/CleaveAndSeq/let7_blocking/final_analysis/final_single_exponential_partial_series_final_norm_final_initialization/figures/'
        normalizationSequence = expTable[expTable[3].str.contains('10')]
        normalizationSequence = normalizationSequence[normalizationSequence[3].str.contains('11')]
        normalizationSequence = normalizationSequence[normalizationSequence[3].str.contains('12')]
        normalizationSequence = normalizationSequence.index[normalizationSequence[3].str.contains('13')].tolist()

        normalizationSequence = normalizationSequence + ['AAAAAACTATACAACCATGTACCTCAAAAAA',
                            'AAAAAACTATACAACCATGAACCTCAAAAAA',
                            'AAAAAACTATACAACCATGATCCTCAAAAAA',
                            'AAAAAACTATACAACCATGATGCTCAAAAAA',
                            'AAAAAACTATACAACCATGATGGTCAAAAAA',
                            'AAAAAACTATACAACCATGATGGACAAAAAA']

    expTable = pd.DataFrame.from_csv(variantTableFilename, sep='\t', header=None, index_col=0)

    # Generate series of counts
    seriesCountTable = generateSeries(seriesList, fastqPrefix, expTableName, experimentName)
    
    normalizedCountTable = normalizeDataSeries(seriesCountTable, normalizationSequence, numTimePoints, expTableName, experimentName)
    normalizedCountTable = pd.read_table(expTableName + 'NormalizedSeriesCounts'+experimentName+'.tsv', sep = '\t')

    # Fit all normalized series
    allresults = fitData(timePoints, numTimePoints, figurelocation, experimentName, expTableName, normalizedCountTable)



#############################################
#############################################
# Run main
miR21 = False
let7 = True

main(miR21, let7)







