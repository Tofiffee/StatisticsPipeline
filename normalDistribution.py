import scipy.stats as stats
import pandas as pd
import numpy as np

def TestNormalDistribution(Data, name):
    if isinstance(Data, pd.DataFrame):
        Data = Data.to_numpy()
    elif isinstance(Data, np.ndarray):
        pass
    else:
        raise ValueError('Data1 and Data2 have to be either pd.DataFrame with 1 column or np.ndarray')

    testData = (Data-Data.mean())/Data.std()

    statistic1, p_shapiro = stats.shapiro(Data)
    statistic2, p_ks = stats.kstest(testData, 'norm')
    andersonResult = stats.anderson(Data, dist='norm')

    if p_shapiro >= 0.05 and p_ks >= 0.05 and andersonResult[0] <= andersonResult[1][3]:
        result = True
        print(name)
        return f'{result} \n' 

    elif p_shapiro < 0.05 and p_ks < 0.05 and andersonResult[0] > andersonResult[1][3]:
        result = False
        print(name)
        return f'{result} \n'

    elif p_ks >= 0.05 and andersonResult[0] <= andersonResult[1][3]:
        result = True
        print(name)
        return f'{result} \n'
    
    elif p_shapiro >= 0.05 and p_ks >= 0.05:
        result = True
        print(name)
        return f'{result} \n'

    else:
        print(name)
        print(f'The Shapiro-Wilk test returns p = {p_shapiro}')
        print(f'The Kolmogorov-Smirnov return p = {p_ks}')
        print(f'The Anderson Test return a statistic of {andersonResult[0]} and a critival value of {andersonResult[1][3]}')