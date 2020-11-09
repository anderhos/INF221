import numpy as np
import pandas as pd
import timeit
import copy

# arguments: algo, seed, input ordering, input_size

# Input size is input_base raised by input_power

def run_benchmark(sort_func, input_power, seed=None, save=True,
                  input_base=2, num_runs=5):
    """
    Run benchmark with given parameters

    Parameters
    ----------
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
    """

    results = pd.DataFrame(columns = ['input order', 'input size',
                                      'run number', 'sorting algorithm',
                                      'time'])
    for order in ['sorted', 'reversed', 'random']:
        for p in range(input_power):

            # Generate random data
            rng = np.random.default_rng(seed)
            test_data = np.random.uniform(size=input_base**p)

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

            print(f"Sorting average time on {order} \
                  data of size {input_base}^{p}:", np.mean(t) / n_ar)

            for run_number in range(num_runs):
                results = \
                    results.append(
                        {'input order': order,
                         'input size': input_base ** input_power,
                         'run number': run_number + 1,
                         'sorting algorithm': f'{sort_func}',
                         'time': t[run_number] / n_ar},
                        ignore_index=True)

