import numpy as np
import math
from itertools import chain
import logging
import random


class TargetBinaryEncoder:
    def __init__(self, y=None, n_intervals=3, add_first_lenghts=tuple(), add_last_lenghts=tuple()):
        """
        Create object of target binary encoder.

        Example of usage:
        ```
        _y = np.array([0,0,0,0,1,1,1,1,1,1,1,0,0,0,0])

        be = TargetBinaryEncoder(y=_y, n_intervals=3, add_first_lenghts=(1,2,), add_last_lenghts=(2,1))

        be.get_y_by_mask([False,False,True,True,True,False,False])
        results:
        ```
        array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
        ```

        another call of same object:
        ```
        be.get_y_by_mask([True,False,True,True,True,False,True])
        ```
        results:
        ```
        array([0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0.])
        ```

        :param y: 1D np.array, presorted in manner entity_id,time. For example, for aircraft data should be sorted: AC,time
        :param n_intervals: int, number of intervals to split sequence of ones to
        :param add_first_lenghts: tuple with integers, added intervals lenghts on left side
        :param add_last_lenghts: tuple with integers, added intervals lenght on right side
        """
        assert type(y) is np.ndarray, "y should be np.array"
        assert len(y.shape) == 1, "y should be 1D np.array"
        assert len(add_first_lenghts) == 0 or np.all([True if type(v) is int and v>0 else False for v in add_first_lenghts]), \
            "add_first_lenghts should be tuple with positive int values"
        assert len(add_last_lenghts) == 0 or np.all(
            [True if type(v) is int and v > 0 else False for v in add_last_lenghts]), \
            "add_first_lenghts should be tuple with positive int values"

        self._n_intervals = n_intervals
        self._indicator_indexes = list()

        last_y = 0
        for ind, cur_y in enumerate(y):
            if last_y and not cur_y:
                self._indicator_indexes.append(ind)
            last_y = cur_y

        self._y_lenghts = list()

        for indicator_index in self._indicator_indexes:

            current_len = 0
            current_position = indicator_index - 1

            while y[current_position] == 1:
                current_len += 1
                current_position -= 1

            self._y_lenghts.append(current_len)
        self._add_first_lenghts = add_first_lenghts
        self._add_last_lenghts = add_last_lenghts
        self._y = y.copy()
        self._binary_size = (self._n_intervals + len(self._add_first_lenghts) + len(self._add_last_lenghts)) * len(
            self._indicator_indexes)

    def get_binary_size(self):
        """
        Get size of binary vector needed for y creation
        :return: int, size of binary vector needed for y creation
        """
        return self._binary_size

    def _split_for_n(self, len_to_split, n):
        to_split = len_to_split
        while to_split != 0:
            cur_result = math.ceil(to_split / n)
            yield cur_result
            n = n - 1
            to_split = to_split - cur_result

    def get_y_by_mask(self, binmask):
        """
        Get y vector for binmask.
        :param binmask: 1D np.ndarray with binary values (True,False)
        :return:
        """
        assert len(binmask)==self.get_binary_size(), "len of binary verctor should be exact as in y_fitter: {}".format(self.get_binary_size())

        y_template = np.zeros(len(self._y))
        binmask_subsequences = [binmask[i:i + self._n_intervals +
                                          len(self._add_first_lenghts) +
                                          len(self._add_last_lenghts)] for i in
                                range(0, len(binmask), self._n_intervals +
                                      len(self._add_first_lenghts) +
                                      len(self._add_last_lenghts))]

        for indicator_index, cur_y_len, binmask_subsequence in zip(self._indicator_indexes,
                                                                   self._y_lenghts,
                                                                   binmask_subsequences):
            # reverse way for added in back
            total_start_position = indicator_index

            for n_to_add, reverse_bin_val in zip(self._add_last_lenghts,
                                                 binmask_subsequence[-len(self._add_last_lenghts):]):
                if reverse_bin_val:
                    y_template[total_start_position:total_start_position + n_to_add] = 1
                total_start_position += n_to_add

            # forward way for added in normal way

            total_start_position = indicator_index
            for cur_ones_len, bin_val in zip(
                    chain(self._split_for_n(cur_y_len, self._n_intervals), self._add_first_lenghts[::-1]),
                    binmask_subsequence[:-len(self._add_last_lenghts)][::-1]):

                if bin_val:
                    y_template[total_start_position - cur_ones_len:total_start_position] = 1

                total_start_position -= cur_ones_len
        return y_template


class BinaryGenetics:
    @staticmethod
    def _do_score_tournament(scores, rounds):
        assert type(rounds) is int, 'number of rounds in tournament should be integer'

        max_ind = random.randint(0, len(scores)-1)

        for r in range(0, max(0, rounds - 1)):
            r_num = random.randint(0, len(scores)-1)
            if scores[r_num] > scores[max_ind]:
                max_ind = r_num
        return max_ind

    @staticmethod
    def _softmax(scores):
        assert np.all(scores>0), 'fitness scores must be positive to use softmax function!'
        return np.array(scores)/np.sum(scores)

    @staticmethod
    def _get_f_beta(v1,v2,beta):
        return (1+beta*beta)*v1*v2/(beta*beta*v1 + v2)

    def __init__(self,
                 n_samples=100,
                 n_generations=20,
                 save_best_n=1,
                 p_point_mutate=0.1,
                 top_n=None,
                 binary_shape=50,
                 inbreed_prob=0.1,
                 random_interchange_prob=0.1,
                 n_interchange_points=1,
                 tournament_rounds=2,
                 cached=True,
                 inbreed_distance_func=None,
                 diversity_distance_func=None,
                 diversity_to_fitness_f_beta=0,
                 logging_obj=None,
                 tqdm_obj=None):
        """
        BinaryGenetics optimizer class
        :param n_samples: number of samples in one generation
        :param n_generations: number of generations to run
        :param save_best_n: number of top species to be saved as-is from current generation
        :param p_point_mutate: probability to have at least one mutation in child after born
        :param binary_shape: shape of binary vector to optimize
        :param inbreed_prob: probability with which second parent is picked from most "closest" by
        inbreed_distance_func from samples
        :param random_interchange_prob: probability of non crossover child production, but when child randomly
        (with prob 0.5) gets parent genes
        :param tournament_rounds: int number of rounds in tournament. if 1 - then parents are selected randomly,
        if more, then algorithm become more selective
        :param n_interchange_points: number of crossover points
        :param cached: use internal caching for eval function
        :param inbreed_distance_func: function to check distance between two binary vectors for distance between them, l1 by default
        :param diversity_distance_fun: function to check diversity between two samples. Greater diversity => greater value (cosine by default)
        :param diversity_to_fitness_f_beta: f-value to take between softmaxed fitness and diversity values,
        :param logging_obj: object, which is used for logging
        :param tqdm_obj: object used for progress bar  painting. If None, then no progress bar created
        """

        self.tournament_rounds = tournament_rounds
        self.diversity_to_fitness_f_beta = diversity_to_fitness_f_beta
        self.n_interchange_points = n_interchange_points
        self.random_interchange_prob = random_interchange_prob
        self.n_samples = n_samples
        self.n_generations = n_generations
        self.save_best_n = save_best_n
        self.p_point_mutate = 1-np.power((1-p_point_mutate),1.0/binary_shape)
        if top_n is None:
            self.top_n = max(save_best_n+1,int(self.n_samples*0.2))
        else:
            self.top_n = max(save_best_n+1, int(top_n))
        self.best_samples = None
        self.best_scores = None
        self.binary_shape = binary_shape
        self._for_choice = [i for i in range(binary_shape)]
        self._cache = dict()
        self._cached = cached
        self._current_samples_hashes = set()
        self._softmax_best_scores = None

        # inbreed
        self.inbreed_prob = inbreed_prob
        self.inbreed_func = inbreed_distance_func
        self.inbreed_rank_probs = [np.exp(-1 - (1 / 10) * (x_)) for x_ in range(self.n_samples + 1)]

        if inbreed_distance_func is None:
            def default_inbreed_func(x, y):
                # l1 default
                return np.sum(np.abs(x.astype(np.int8) - y.astype(np.int8)))

            self.inbreed_func = default_inbreed_func

        if diversity_distance_func is None:
            def default_diversity_distance_func(x, y):
                # default diversity distance is cosine distance
                x_int = x.astype(np.int8)
                y_int = y.astype(np.int8)
                return x_int.dot(y_int)/(np.sqrt(np.sum(x_int)*np.sum(y_int)))

            self.diversity_distance_func = default_diversity_distance_func

        if logging_obj is None:
            self.logging_obj = logging.info
        else:
            self.logging_obj = logging_obj
        self.tqdm_obj = tqdm_obj
        if self.tqdm_obj is None:
            self.tqdm_obj = lambda x: x

    def set_eval_func(self, eval_func, greater_is_better=True):
        '''
        Set evaluation func for binary vector
        :param eval_func: function with one argument: bunary vector. Function must return only one float number:
        quality (or score) of a child
        :param greater_is_better: if True, then optimization is targeted for maximizing
        :return: None
        '''
        self.eval_func = eval_func
        self.greater_is_better = greater_is_better

    def _call_eval_func(self, sample):
        if self._cached:
            sample_tuple = tuple([bool(s) for s in sample])
            if sample_tuple in self._cache:
                return self._cache[sample_tuple]
            value = self.eval_func(sample)
            self._cache[sample_tuple] = value
            return value
        return self.eval_func(sample)

    def _is_in_pop(self, sample):
        sample_tuple = tuple([bool(s) for s in sample])
        return sample_tuple in self._current_samples_hashes

    def _add_to_pop(self, sample):
        self._current_samples_hashes.add(tuple([bool(s) for s in sample]))

    def _clear_pop(self):
        self._current_samples_hashes.clear()

    def _do_diversity_scores(self,samples):
        result = np.zeros(len(samples))
        int_samples = samples.astype(np.int8)
        for i,sample in enumerate(int_samples):
            sample_score = 0

            for j,other_sample in enumerate(int_samples):
                if i != j:
                    sample_score = self.diversity_distance_func(sample,other_sample)

            result[i] = sample_score/(len(samples)-1)
        return result

    def learn(self, yield_best=False, samples=None):
        '''
        Main method to start optimizing
        :param yield_best: create iterator fot best sample in generation
        :param samples: initial samples to start from
        :return: iterator, if yield_best is set to True, else list of best samples with according
        scores from each generation
        '''
        # initial samples
        if samples is None:
            samples = np.random.random((self.n_samples, self.binary_shape)) > 0.5

        current_gen = 0
        prev_saved = None
        while current_gen < self.n_generations:
            if type(prev_saved) is list:
                scores = np.array(prev_saved + [self._call_eval_func(samples[i]) for i in
                                                self.tqdm_obj(range(len(prev_saved), samples.shape[0]))])
                prev_saved.clear()
            else:
                scores = np.array([self._call_eval_func(samples[i]) for i in
                                   self.tqdm_obj(range(samples.shape[0]))])  # self.eval_func, 1, samples)
                prev_saved = list()


            if self.greater_is_better:
                best_ranks = np.argsort(scores)[::-1][:self.top_n]
            else:
                best_ranks = np.argsort(scores)[:self.top_n]
            sorted_scores = scores[best_ranks]

            self.logging_obj('Generation {} top score {} median score {}'.format(current_gen, sorted_scores[0],
                                                                                 np.median(sorted_scores)))

            self.best_scores = sorted_scores
            self._softmax_best_scores = self._softmax(sorted_scores)
            if self.best_samples is None:
                self._diversity_scores = self._do_diversity_scores(samples)
            else:
                self._diversity_scores = self._do_diversity_scores(self.best_samples)

            self._total_f_beta_scores = [self._get_f_beta(v1,v2,self.diversity_to_fitness_f_beta)
                                         for v1,v2 in zip(self._softmax_best_scores,self._diversity_scores)]

            self.best_samples = samples[best_ranks, :]

            if yield_best:
                yield self.best_samples[0], self.best_scores[0]

            current_samples = 0
            samples = np.zeros((self.n_samples, self.binary_shape))

            while current_samples < self.n_samples:
                if current_samples < self.save_best_n:
                    samples[current_samples,:] = self.best_samples[current_samples]
                    self._add_to_pop(samples[current_samples])
                    prev_saved.append(self.best_scores[current_samples])
                    current_samples += 1
                    continue

                # do inbreed if probability to do so
                if np.random.random() < self.inbreed_prob:
                    # inbreed here
                    first_index = self._do_score_tournament(self.best_scores,self.tournament_rounds)
                    first_sample = self.best_samples[first_index]

                    # inbreed probability is not based on score, but on other "inbreed_func" value
                    other_scores = np.array([self.inbreed_func(first_sample, sample) for sample in self.best_samples])
                    second_index = self._do_score_tournament(other_scores,self.tournament_rounds)

                else:
                    # random choice acc to scores
                    # first_index = self._do_score_tournament(self.best_scores, self.tournament_rounds)
                    # second_index = self._do_score_tournament(self.best_scores, self.tournament_rounds)
                    first_index = self._do_score_tournament(self._total_f_beta_scores, self.tournament_rounds)
                    second_index = self._do_score_tournament(self._total_f_beta_scores, self.tournament_rounds)

                if np.random.random() < self.random_interchange_prob:
                    # random x-y interchange
                    for v in range(self.binary_shape):
                        samples[current_samples, v] = self.best_samples[first_index, v] if np.random.random() > 0.5 else \
                        self.best_samples[second_index, v]
                else:
                    # point-cross interchange
                    from_first_parent = True
                    interchange_points = set(
                        [random.choice(self._for_choice) for i in range(self.n_interchange_points)])

                    for v in range(self.binary_shape):
                        if v in interchange_points:
                            from_first_parent = not from_first_parent

                        samples[current_samples, v] = self.best_samples[first_index, v] if from_first_parent else \
                        self.best_samples[second_index, v]

                samples[current_samples] = self.mutate_sample(samples[current_samples])# if sample is already in population

                if self._is_in_pop(samples[current_samples]):
                    continue

                self._add_to_pop(samples[current_samples])
                current_samples += 1

            self._clear_pop()
            samples = samples.astype(bool)
            current_gen += 1

        return self.best_samples, self.best_scores

    def mutate_sample(self, sample):
        '''
        # Mutate sample. Probability to get non mutated is (1-self.p_point_mutate)
        :param sample:
        :return:
        '''
        for mutation_index,v in enumerate(sample):
            if np.random.random() < self.p_point_mutate:
                sample[mutation_index] = not v
        return sample


def check_iter(some_obj):
    try:
        _ = iter(some_obj)
        return True
    except TypeError as te:
        return False


def unpack_result_from_tuple(some_func):
        def some_new_func(*args, **kwargs):
            result = some_func(*args, **kwargs)
            if check_iter(result):
                if len(result)==1:
                    return result[0]
            return result

        return some_new_func


class BinGoldenSearch():
    def __init__(self,
                 grid_to_search,
                 trial_func=None,
                 trial_result_order_func=None,
                 trial_result_distance_func=None,
                 computed_trials_info=None):

        self._grid_to_search=list()
        for v in grid_to_search:
            if check_iter(v):
                self._grid_to_search.append(tuple(v))
            else:
                self._grid_to_search.append(tuple((v,)))


        if computed_trials_info:
            self._computed_trials_info = computed_trials_info
        else:
            self._computed_trials_info = dict()

        if trial_func:
            self._trial_func = trial_func
        else:
            self._trial_func = None

        if trial_result_order_func:
            self._trial_result_order_func = trial_result_order_func
        else:
            self._trial_result_order_func = lambda x, y: x[0] < y[0]

        if trial_result_distance_func:
            self._trial_result_distance_func = trial_result_distance_func
        else:
            self._trial_result_distance_func = lambda x, y: abs(x[0] - y[0])

    def set_trial_func(self, some_func):
        self._trial_func = some_func

    def compute_trial(self,args_index):
        args = self._grid_to_search[args_index]

        # return if already computed
        if args in self._computed_trials_info:
            return self._computed_trials_info[args]

        # compute and memoize
        trial_result = self._trial_func(*args)
        if not check_iter(trial_result):
            trial_result = tuple((trial_result,))

        self._computed_trials_info[args] = trial_result
        return trial_result

    @unpack_result_from_tuple
    def do_search(self, val_to_find,min_index=-1,max_index=-1):
        ########## special cases here #############

        # nothing to search
        if len(self._grid_to_search) == 0:
            return None

        # search space contains 1 argument
        if len(self._grid_to_search) == 1:
            return self._grid_to_search[0]

        # we're ended search - only 1 last element to try
        if (min_index == max_index) and (min_index!=-1):
            return self._grid_to_search[min_index]

        if not check_iter(val_to_find):
            val_to_find = tuple((val_to_find,))

        # we're ended search - only 2 last elements to try
        if max_index-min_index == 1:
            min_result = self.compute_trial(min_index)
            max_result = self.compute_trial(max_index)
            if self._trial_result_distance_func(val_to_find,min_result)<self._trial_result_distance_func(val_to_find,max_result):
                return self._grid_to_search[min_index]
            else:
                return self._grid_to_search[max_index]

        ########## non special case ###############
        # initialization of search, only first time
        if (min_index == -1) and (max_index == -1):
            min_index = 0
            max_index = len(self._grid_to_search) - 1

        # cur index is in center
        cur_index = int((max_index+min_index)/2)
        trial_result = self.compute_trial(cur_index)

        if self._trial_result_order_func(val_to_find,trial_result):
            return self.do_search(val_to_find,min_index=min_index,max_index=cur_index)

        return self.do_search(val_to_find, min_index=cur_index, max_index=max_index)