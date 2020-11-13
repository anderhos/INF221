import numpy as np
import pandas as pd
import timeit
import copy
import os.path


def run_benchmark(sort_func, input_base, input_power, seed=None, save=True,
                  num_runs=5):
    """
    Run benchmark with given parameters

    Parameters
    ----------------------------------------------------------------------
    sort_func: function
               Algorithm to used for sorting.
    input_power: int
                 The power of the data size
    seed: int, optional
    save: bool, optional
          Saving results to file. Default is True.
    input_base: int, optional
                Raise this number to input_power to determine data size
    num_runs: int, optional
              Number of runs at each test case
    ----------------------------------------------------------------------
    """

    input_size = input_base**input_power
    # Create data frame for storing results
    results = pd.DataFrame(columns = ['input order', 'input size',
                                      'run number', 'sorting algorithm',
                                      'time'])
    for order in ['sorted', 'reversed', 'random']:
        for p in range(input_power+1):

            # Generate random data
            rng = np.random.default_rng(seed)
            test_data = rng.uniform(size=input_base**p)

            # Presorting
            if order == 'sorted':
                test_data = sorted(test_data)

            elif order == 'reversed':
                test_data = list(reversed(sorted(test_data)))

            # Timer function
            clock = timeit.Timer(stmt='sort_func(copy(data))',
                                 globals={'sort_func': sort_func,
                                          'data': test_data,
                                          'copy': copy.copy})
            n_ar, t_ar = clock.autorange()
            t = clock.repeat(repeat=5, number=n_ar)

            # Print out average time over the number of runs for each data size
            print(f"Minimum time(s) on {order} data of size "
                  f"{input_base**p}:", np.min(t) / n_ar)

            for run_number in range(num_runs):
                results = \
                    results.append(
                        {'input order': order,
                         'input size': input_base**p,
                         'run number': run_number + 1,
                         'sorting algorithm': f'{sort_func.__name__}',
                         'time': t[run_number] / n_ar},
                        ignore_index=True)

    if save:
        # Save pickled data frame to file in data directory
        directory = '../data/'
        filename = '{0}_n{1}.pkl'.format(sort_func.__name__, input_size)
        file_path = os.path.join(directory, filename)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        results.to_pickle(file_path)
        print()
        print(f'Saved to path: {file_path}')