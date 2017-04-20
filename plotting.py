%matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import timing


def trans_length_plot(filename):
    data = timing.load(filename)
    fp = data['fp_growth']
    apriori = data['apriori']
    transaction_indices = fp['transactions']
    fp_times = fp[:, 0, 0, 0, 0, 0]
    apriori_times = apriori[:, 0, 0, 0, 0, 0]
    transaction_lengths = [
        timing.get_run_params(x, 0, 0, 0, 0, 0, data).transactions
        for x in transaction_indices]
    frame = pd.DataFrame()
    frame['transactions'] = transaction_lengths
    frame['apriori'] = apriori_times.values
    frame['fp'] = fp_times.values
    frame = frame.set_index('transactions')
    frame.plot()
