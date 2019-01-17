import collections
import itertools
import numpy as np
import sys

# Import optional dependencies
_PANDAS_AVAILABLE = False
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pass
_STATS_AVAILABLE = False
try:
    from scipy import stats
    _STATS_AVAILABLE = True
except ImportError:
    sys.stderr.write('Cannot import scipy.  Some models may not be available.\n')
    sys.stderr.flush()
    pass

_FLOAT_EQ_DELTA = 0.000001  # For comparing float equality


class SValueModel:
    """Model implementing SVA."""


    def compute_outlier_scores(self, frequencies):
        """Computes the SVA outlier scores fo the given frequencies dictionary.
        
        Args:
            frequencies: dictionary of dictionaries, mapping (aggregation unit) -> (value) ->
                (number of times aggregation unit reported value).
            
        Returns:
            dictionary mapping (aggregation unit) -> (SVA outlier score for aggregation unit).
        """
        if (len(frequencies.keys()) < 2):
            raise Exception("There must be at least 2 aggregation units.")
        outlier_values = {}
        rng = list(frequencies[list(frequencies.keys())[0]].keys())
        normalized_frequencies = {}
        for j in frequencies.keys():
            # If j doesn't have any answers for given question, remove j and
            # assign outlier score of 0.
            if (sum(frequencies[j].values()) == 0):
                del frequencies[j]
                outlier_values[j] = 0
                continue
            normalized_frequencies[j] = _normalize_counts(frequencies[j])
        medians = {}    
        for r in rng:
            medians[r] = np.median([normalized_frequencies[j][r]
                for j in normalized_frequencies.keys()])
        for j in frequencies.keys():
            outlier_values[j] = 0
            for r in rng:
                outlier_values[j] += abs(normalized_frequencies[j][r] - medians[r])
        return self._normalize(outlier_values)
    
    
    def _normalize(self, value_dict):
        """Divides everything in value_dict by the median of values.

        If the median is less than 1 / (# of aggregation units), it divides everything by
        (# of aggregation units) instead.
        
        Args:
            value_dict: dictionary of the form (aggregation unit) -> (value).
        Returns:
            dictionary of the same form as value_dict, where the values are normalized as described
            above.
        """
        median = np.median([value_dict[i] for i in value_dict.keys()])
        n = len(value_dict.keys())
        if median < 1.0 / float(n):
            divisor = 1.0 / float(n)
        else:
            divisor = median
        return_dict = {}
        for i in value_dict.keys():
            return_dict[i] = float(value_dict[i]) / float(divisor)
        return return_dict


########################################## Helper functions ########################################

def _normalize_counts(counts, val=1):
    """Normalizes a dictionary of counts, such as those returned by _get_frequencies().

    Args:
        counts: a dictionary mapping value -> count.
        val: the number the counts should add up to.
    
    Returns:
        dictionary of the same form as counts, except where the counts have been normalized to sum
        to val.
    """
    n = sum(counts.values())
    frequencies = {}
    for r in counts.keys():
        frequencies[r] = val * float(counts[r]) / float(n)
    return frequencies


def _get_frequencies(data, col, col_vals, agg_col, agg_unit, agg_to_data):
    """Computes a frequencies dictionary for a given column and aggregation unit.
    
    Args:
        data: numpy.recarray or pandas.DataFrame containing the data.
        col: name of column to compute frequencies for.
        col_vals: a list giving the range of possible values in the column.
        agg_col: string giving the name of the aggregation unit column for the data.
        agg_unit: string giving the aggregation unit to compute frequencies for.

        agg_to_data: a dictionary of aggregation values pointing to subsets of data
    Returns:
        A dictionary that maps (column value) -> (number of times agg_unit has column value in
        data).
    """
    interesting_data = None
    frequencies = {}
    for col_val in col_vals:
        frequencies[col_val] = 0
        # We can't just use collections.Counter() because frequencies.keys() is used to determine
        # the range of possible values in other functions.
    if _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        interesting_data = agg_to_data[agg_unit][col]
        for name in interesting_data:
            if name in frequencies:
                frequencies[name] = frequencies[name] + 1
    else:  # Assumes it is an np.ndarray
        for row in itertools.ifilter(lambda row : row[agg_col] == agg_unit, data):
            if row[col] in frequencies:
                frequencies[row[col]] += 1
    return frequencies, interesting_data

def _run_alg(data, agg_col, cat_cols, model, null_responses):
    """Runs an outlier detection algorithm, taking the model to use as input.
    
    Args:
        data: numpy.recarray or pandas.DataFrame containing the data.
        agg_col: string giving the name of aggregation unit column.
        cat_cols: list of the categorical column names for which outlier values should be computed.
        model: object implementing a compute_outlier_scores() method as described in the comments
            in the models section.
        null_responses: list of strings that should be considered to be null responses, i.e.,
            responses that will not be included in the frequency counts for a column.  This can
            be useful if, for example, there are response values that mean a question has been
            skipped.
    
    Returns:
        A dictionary of dictionaries, mapping (aggregation unit) -> (column name) ->
        (outlier score).
    """
    agg_units = sorted(set(data[agg_col]))
    outlier_scores = collections.defaultdict(dict)
    agg_to_data = {}
    agg_col_to_data = {}
    for agg_unit in agg_units:
        # TODO: could this be smarter and remove data each time? maybe no savings.
        # TODO: support numpy only again
        agg_to_data[agg_unit] = data[data[agg_col] == agg_unit]
        agg_col_to_data[agg_unit] = {}
        
    for col in cat_cols:
        col_vals = sorted(set(data[col]))
        col_vals = [c for c in col_vals if c not in null_responses]
        frequencies = {}
        for agg_unit in agg_units:
            frequencies[agg_unit],grouped = _get_frequencies(data, col, col_vals, agg_col, agg_unit, agg_to_data)
            agg_col_to_data[agg_unit][col] = grouped
        outlier_scores_for_col = model.compute_outlier_scores(frequencies)
        for agg_unit in agg_units:
            outlier_scores[agg_unit][col] = outlier_scores_for_col[agg_unit]
    return outlier_scores, agg_col_to_data


########################################## Public functions ########################################

# if _STATS_AVAILABLE:
#     def run_mma(data, aggregation_column, categorical_columns, null_responses=[]):
#         """Runs the MMA algorithm (requires scipy module).
        
#         Args:
#             data: numpy.recarray or pandas.DataFrame containing the data.
#             aggregation_column: a string giving the name of aggregation unit column.
#             categorical_columns: a list of the categorical column names for which outlier values
#                 should be computed.
#             null_responses: list of strings that should be considered to be null responses, i.e.,
#                 responses that will not be included in the frequency counts for a column.  This can
#                 be useful if, for example, there are response values that mean a question has been
#                 skipped.
        
#         Returns:
#             A dictionary of dictionaries, mapping (aggregation unit) -> (column name) ->
#             (mma outlier score).
#         """
#         return _run_alg(data,
#                         aggregation_column,
#                         categorical_columns,
#                         MultinomialModel(),
#                         null_responses)

def run_sva(data, aggregation_column, categorical_columns, null_responses=[]):
    """Runs the SVA algorithm.
        
    Args:
        data: numpy.recarray or pandas.DataFrame containing the data.
        aggregation_column: a string giving the name of aggregation unit column.
        categorical_columns: a list of the categorical column names for which outlier values
            should be computed.
        null_responses: list of strings that should be considered to be null responses, i.e.,
            responses that will not be included in the frequency counts for a column.  This can
            be useful if, for example, there are response values that mean a question has been
            skipped.
        
    Returns:
        A dictionary of dictionaries, mapping (aggregation unit) -> (column name) ->
        (sva outlier score).
    """
    return _run_alg(data,
                    aggregation_column,
                    categorical_columns,
                    SValueModel(),
                    null_responses)
