"""Compare runtimes of apriori and fp growth algorithms.

Comparisons are run across datasets of varying characteristics.

Compare to https://arxiv.org/abs/1701.09042
"""
import collections
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import xarray as xr

import apriori
import dataset
import fpgrowth

MAX_TIME = 1800  # Cap each run to 30 minutes
SAVE_FILE = 'itemsize-grid-small.nc'

MIN_TRANSACTIONS = 100
MAX_TRANSACTIONS = 2000
TRANSACTION_STEPS = 10
MIN_SUPPORT = .02
MAX_SUPPORT = .4
SUPPORT_STEPS = 1
MIN_ITEMS = 20
MAX_ITEMS = 300
ITEMS_STEPS = 3
MIN_BASKET_SIZE = 15
MAX_BASKET_SIZE = 50
BASKET_STEPS = 1
MIN_LIKELY = 5
LIKELY_STEPS = 1
LIKELY_FRACTION_START = .2
LIKELY_FRACTION_STEPS = 3

# MIN_TRANSACTIONS = 10000
# MAX_TRANSACTIONS = 2000000
# TRANSACTION_STEPS = 10
# MIN_SUPPORT = .02
# MAX_SUPPORT = .4
# SUPPORT_STEPS = 1
# MIN_ITEMS = 20
# MAX_ITEMS = 300
# ITEMS_STEPS = 3
# MIN_BASKET_SIZE = 15
# MAX_BASKET_SIZE = 50
# BASKET_STEPS = 1
# MIN_LIKELY = 5
# LIKELY_STEPS = 1
# LIKELY_FRACTION_START = .2
# LIKELY_FRACTION_STEPS = 3

Parameters = collections.namedtuple('Parameters',
                                    ['transactions', 'support', 'basket',
                                     'items', 'likely', 'likely_fraction'])


def timed_itemset_mining(cls, transactions, support, results, name):
    start = timer()
    cls(transactions, support)
    stop = timer()
    results[name] = stop - start
    return stop - start


def construct_dataset():
    shape = [TRANSACTION_STEPS, SUPPORT_STEPS, BASKET_STEPS, ITEMS_STEPS,
             LIKELY_STEPS, LIKELY_FRACTION_STEPS]
    ap_data = xr.DataArray(
        np.zeros(shape), dims=['transactions', 'support',  'basket',
                               'items', 'likely', 'likely_fraction'])
    fp_data = xr.DataArray(
        np.zeros(shape), dims=['transactions', 'support',
                               'basket', 'items', 'likely', 'likely_fraction'])
    attrs = dict(
        min_transactions=MIN_TRANSACTIONS,
        max_transactions=MAX_TRANSACTIONS,
        transaction_steps=TRANSACTION_STEPS,
        min_support=MIN_SUPPORT,
        max_support=MAX_SUPPORT,
        support_steps=SUPPORT_STEPS,
        min_items=MIN_ITEMS,
        max_items=MAX_ITEMS,
        items_steps=ITEMS_STEPS,
        min_basket_size=MIN_BASKET_SIZE,
        max_basket_size=MAX_BASKET_SIZE,
        basket_steps=BASKET_STEPS,
        min_likely=MIN_LIKELY,
        likely_steps=LIKELY_STEPS,
        likely_fraction_start=LIKELY_FRACTION_START,
        likely_fraction_steps=LIKELY_FRACTION_STEPS,
    )
    data = xr.Dataset({'apriori': ap_data, 'fp_growth': fp_data}, attrs=attrs)
    return data


def save(data, filename):
    data.to_netcdf(filename)


def load(filename):
    data = xr.open_dataset(filename)
    data.load()
    data.close()
    return data


def index_iterator(data):
    for t in range(data.attrs['transaction_steps']):
        print('Num transactions: {} of {}'
              .format(t+1, data.attrs['transaction_steps']))
        for s in range(data.attrs['support_steps']):
            for b in range(data.attrs['basket_steps']):
                print('Basket index: {} of {}'
                      .format(b+1, data.attrs['basket_steps']))
                for i in range(data.attrs['items_steps']):
                    for l in range(data.attrs['likely_steps']):
                        for lf in range(data.attrs['likely_fraction_steps']):
                            yield (t, s, b, i, l, lf)


def get_run_params(t, s, b, i, l, lf, data):
    transactions = translate(data.attrs['min_transactions'],
                             data.attrs['max_transactions'],
                             t, data.attrs['transaction_steps'])
    support = int(transactions * translate(
        data.attrs['min_support'], data.attrs['max_support'], s,
        data.attrs['support_steps'], lambda x: x))
    basket = translate(data.attrs['min_basket_size'],
                       data.attrs['max_basket_size'], b,
                       data.attrs['basket_steps'])
    items = translate(data.attrs['min_items'],
                      data.attrs['max_items'], i,
                      data.attrs['items_steps'])
    likely = translate(data.attrs['min_likely'], items-2, l,
                       data.attrs['likely_steps'])
    d = likely / items
    d = d + (1 - d) * data.attrs['likely_fraction_start']
    likely_fraction = d + (1 - d) * lf / data.attrs['likely_fraction_steps']
    return Parameters(transactions, support, basket, items, likely,
                      likely_fraction)


def generate_transactions(params):
    return dataset.construct(params.transactions, params.basket,
                             params.items, params.likely,
                             params.likely_fraction)


def translate(min, max, index, num_steps, convert=int):
    return convert(min + (max - min) * index / num_steps)


def time_different_dims(data, max_time=10):
    for indices in index_iterator(data):
        jobs = []
        params = get_run_params(*indices, data)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        T = generate_transactions(params)
        fp_process = multiprocessing.Process(
            target=timed_itemset_mining,
            args=(fpgrowth.FpGrowth, T, params.support, return_dict, 'fp'))
        jobs.append(('fp', fp_process))
        apriori_process = multiprocessing.Process(
            target=timed_itemset_mining,
            args=(apriori.Apriori, T, params.support, return_dict, 'ap'))
        jobs.append(('ap', apriori_process))
        fp_process.start()
        apriori_process.start()
        apriori_process.join(max_time)
        for name, job in jobs:
            if job.is_alive():
                print("Terminating {}".format(name))
                job.terminate()
                job.join()
                return_dict[name] = max_time
        data['apriori'][Parameters(*indices)._asdict()] = return_dict['ap']
        data['fp_growth'][Parameters(*indices)._asdict()] = return_dict['fp']


if __name__ == '__main__':
    data = construct_dataset()
    time_different_dims(data, MAX_TIME)
    save(data, SAVE_FILE)
