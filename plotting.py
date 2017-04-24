# %matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

import timing

sns.set_style('darkgrid')


def trans_length_plot(filename, outfile=None):
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
    fig = plt.gcf()
    if outfile:
        fig.savefig(outfile)


def grid_plot(filename, outfile=None):
    data = timing.load(filename)
    fp = data['fp_growth']
    apriori = data['apriori']
    param_list = []
    for index in timing.index_iterator(data):
        for time, name in [(fp, 'fp'), (apriori, 'apriori')]:
            params = timing.get_run_params(*index, data)
            param_d = timing.Parameters(*index)._asdict()
            param_list.append(
                [params.transactions, params.support, params.basket,
                 params.items, index[4], index[5],
                 time[param_d], name])
    frame = pd.DataFrame(
        param_list,
        columns=['transactions', 'support', 'basket',
                 'items', 'likely', 'likely_fraction', 'time', 'algorithm'])
    grid = sns.FacetGrid(frame, col='items', row='likely_fraction', hue='algorithm')
    grid = grid.map(plt.plot, 'transactions', 'time').add_legend()
    fig = plt.gcf()
    if outfile:
        fig.savefig(outfile)
