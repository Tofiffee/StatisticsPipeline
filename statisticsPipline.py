from multiprocessing.sharedctypes import Value
import scipy.stats as stats
import pandas as pd
import numpy as np

def TestNormalDistribution(DataDict):
    if isinstance(DataDict['Data1'], pd.DataFrame):
        Data1 = DataDict['Data1'].to_numpy()
        Data2 = DataDict['Data2'].to_numpy()
    elif isinstance(DataDict['Data1'], np.ndarray):
        Data1 = DataDict['Data1']
        Data2 = DataDict['Data2']
    else:
        raise ValueError('Data1 and Data2 have to be either pd.DataFrame with 1 column or np.ndarray')

    testData1 = (Data1-Data1.mean())/Data1.std()
    testData2 = (Data2-Data2.mean())/Data2.std()

    statistic1, p_shapiro1 = stats.shapiro(DataDict['Data1'])
    statistic3, p_shapiro2 = stats.shapiro(DataDict['Data2'])

    statistic2, p_ks1 = stats.kstest(testData1, 'norm')
    statistic4, p_ks2 = stats.kstest(testData2, 'norm')

    andersonResult1 = stats.anderson(DataDict['Data1'], dist='norm')
    andersonResult2 = stats.anderson(DataDict['Data2'], dist='norm')

    if p_shapiro1 >= 0.05 and p_ks1 >= 0.05 and p_shapiro2 >= 0.05 and p_ks2 >= 0.05 and andersonResult1[0] <= andersonResult1[1][3] and andersonResult2[0] <= andersonResult2[1][3]:
        DataDict['norm'] = True
        return DataDict

    elif p_shapiro1 < 0.05 or p_ks1 < 0.05 or p_shapiro2 < 0.05 or p_ks2 < 0.05 and andersonResult1[0] > andersonResult1[1][3] and andersonResult2[0] > andersonResult2[1][3]:
        DataDict['norm'] = False
        return DataDict

    elif p_ks1 >= 0.05 and p_ks2 >= 0.05 and andersonResult1[0] <= andersonResult1[1][3] and andersonResult2[0] <= andersonResult2[1][3]:
        DataDict['norm'] = True
        return DataDict
    
    elif p_shapiro1 >= 0.05 and p_ks1 >= 0.05 and p_shapiro2 >= 0.05 and p_ks2 >= 0.05:
        DataDict['norm'] = True
        return DataDict

    else:
        print(DataDict['Name'])
        print(f'The Shapiro-Wilk test returns p = {p_shapiro1} for Data1')
        print(f'The Kolmogorov-Smirnov return p = {p_ks1} for Data1')
        print(f'The Anderson Test return a statistic of {andersonResult1[0]} and a critival value of {andersonResult1[1][3]} for Data1')
        print(f'The Shapiro-Wilk test returns p = {p_shapiro2} for Data2')
        print(f'The Kolmogorov-Smirnov return p = {p_ks2} for Data2')
        print(f'The Anderson Test return a statistic of {andersonResult2[0]} and a critival value of {andersonResult2[1][3]} for Data2')
        print('Please dicide for a test stratigy:')
        strategy = input('Choose normal distribution (True/False: ')
        DataDict['norm'] = bool(strategy)

def statisticalTestingDependent(DataDict):
    if DataDict['norm'] == True:
        statistic, p_val = stats.ttest_rel(DataDict['Data1'], DataDict['Data2'], alternative=DataDict['alternativ'])
        name = DataDict['name']
        return f'The students-T-test for dependent dataset returns a p value of {p_val} with a statistic of {statistic} for the datasets {name}\n'
    elif DataDict['norm'] == False:
        statistic, p_val = stats.wilcoxon(DataDict['Data1'], DataDict['Data2'], zero_method='pratt', alternative=DataDict['alternativ'])
        name = DataDict['name']
        return f'The Wilcoxen-Pratt signed rank test returns a p value of {p_val} with a statistic of {statistic} for the datasets {name}\n'

def statisticalTestingIndependent(DataDict: dict):
    if DataDict['norm'] == True:
        statistic, p_lev = stats.levene(DataDict['Data1'], DataDict['Data2'])
        if p_lev >= 0.05:
            statistic, p_val = stats.ttest_ind(DataDict['Data1'], DataDict['Data2'], equal_var=True, alternative=DataDict['alternativ'])
            name = DataDict['name']
            return f'The students-T-test with equal variance returns a p value of {p_val} with a statistic of {statistic}for the datasets {name}\n'
        elif p_lev < 0.05:
            statistic, p_val = stats.ttest_ind(DataDict['Data1'], DataDict['Data2'], equal_var=False, alternative=DataDict['alternativ'])
            name = DataDict['name']
            return f'The students-T-test with no equal variance returns a p value of {p_val} with a statistic of {statistic} for the datasets {name}\n'
    elif DataDict['norm'] == False:
        statistic_man, p_val_man = stats.mannwhitneyu(DataDict['Data1'], DataDict['Data2'], use_continuity=True, alternative=DataDict['alternativ'])
        statistic_ks, p_val_ks = stats.ks_2samp(DataDict['Data1'], DataDict['Data2'], alternative=DataDict['alternativ'])
        name = DataDict['name']
        return f'The MannWhitney-U test returns a p value of {p_val_man} with a statistic of {statistic_man}for the datasets {name}\nThe Kolmogorov-Smirnov test return a p value of {p_val_ks} and a statistic of {statistic_ks} for the datasets {name}\n'


def StatisticalTesting(DataDict: dict) -> str:
    """_summary_

    Args:
        DataDict (dict): _description_

    Returns:
        str: _description_
    """
    DataDict = TestNormalDistribution(DataDict)
    if DataDict['dependent'] == True:
        result = statisticalTestingDependent(DataDict)
        return result
    elif DataDict['dependent'] == False:
        result = statisticalTestingIndependent(DataDict)
        return result