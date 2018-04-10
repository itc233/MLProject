import numpy as np
import random
from multiprocessing import Pool

class GA(object):

    # the last column of train and valid will be taken as the perdict value
    def __init__(self, train, valid, estimator, feval, 
                 groups=100, iter=200, r_sample=0.8, r_crossover=0.5, r_vary=0.01,
                 r_keep_best=0.1, n_jobs=4,
                 verbose=False):
        self.train = train
        self.estimator = estimator
        self.verbose = verbose
        self.iter = iter
        self.groups = groups
        self.r_sample = r_sample
        self.valid = valid
        self.r_crossover = r_crossover
        self.r_vary = r_vary
        self.r_keep_best = r_keep_best
        self.n_jobs = n_jobs

        self.feval = feval
        self._validate()

    def _verbose(self, *args):
        if self.verbose:
            print(*args)

    def _validate(self):
        assert self.train.shape[1] == self.valid.shape[1]
        assert self.iter > 0, 'iteration cnt should > 0'
        assert self.r_sample > 0, 'r_sample is invalid'
        assert self.estimator, 'estimator is invalid'

    # axis = [0,1]: 0 means row, 1 means columns
    def select(self, axis):
        assert axis in [0, 1], 'axis should be 0 or 1'
        if axis == 0:
            return self.select_instance()
        else:
            return self.select_feature()

    # calculate the adaptation score of a sample gene
    # the smaller the better
    def _sample_adapt_score(self, gene):
        sample = self.train.T[:-1][gene].T
        estor = self.estimator.fit(sample, self.train.T[-1].T)
        sample_v = self.valid.T[:-1][gene].T
        predicts = estor.predict(sample_v)
        return self.feval(predicts, self.valid.T[-1].T)

    def select_instance(self):
        n_instance = self.train.shape[0]
        n_sample = int(self.r_sample * n_instance)

        (best_gene, best_scores) = self._run(
            n_instance, n_sample, self._sample_adapt_score)

        best_sample = self.train[best_gene]

        # return samples, best gene, and the vary bests scores
        return (best_sample, best_gene, best_scores)

    # the smaller the better
    def _select_feature_adapt_score(self, gene):
        sample = self.train.T[:-1][gene].T
        estor = self.estimator.fit(sample, self.train.T[-1].T)
        valid_fs = self.valid.T[:-1][gene].T
        predicts = estor.predict(valid_fs)
        return self.feval(predicts, self.valid.T[-1].T)

    def select_feature(self):
        n_features = self.train.shape[1] - 1  # the last feature is the value
        n_sample = int(self.r_sample * n_features)

        (best_gene, best_scores) = self._run(
            n_features, n_sample, self._select_feature_adapt_score
        )

        best_sample = self.train.T[:-1][best_gene].T
        return (best_sample, best_gene, best_scores)

    # generate a random array whose len=n_len, and have n_pos's value==True
    def _random_series(self, n_len, n_pos):
        gene = np.zeros(n_len, dtype=np.bool)

        indexes = np.arange(n_len, dtype=np.int)
        random.shuffle(indexes)

        gene[indexes[:n_pos]] = True
        return gene

    def _gambling_board(self, scores):

        # scores: the smaller the better
        # possbilities: the bigger the better
        # p = (alpha)^score

        alpha = 0.5
        tmp = list(map(lambda x: pow(alpha, x), scores))
        tmp = tmp / np.sum(tmp)               # calculate p
        possbilities = tmp / np.min(tmp)      # scale
        gambling_board = []
        tmp_array = []
        #for i in range(len(possbilities)):
        #    tmp_array.concatenate((tmp_array, np.full(int(possbilities[i]), i, dtype=np.int))
        #random.shuffle(gambling_board)
        return possbilities.astype(int)#gambling_board

    # generate children
    def _reproduce(self, gene1, gene2):
        if np.array_equal(gene1, gene2):
            return gene1

        # crossover
        children = gene1.copy()

        co_flags = self._random_series(
            len(gene2), int(len(gene2) * self.r_crossover))
        children[co_flags] = gene2[co_flags]

        # variation
        va_flags = self._random_series(len(gene2), int(len(gene2) * self.r_vary))
        children[va_flags] = ~children[va_flags]

        return children.tolist()

    def _reproduce_wrapper(self, genes, board, bid1, bid2):
        gene1 = genes[board[bid1]]
        gene2 = genes[board[bid2]]

        return self._reproduce(gene1, gene2)
    
    def _run(self, n_gene_units, n_sample, adapt_func):
        if n_gene_units == n_sample:
            return self.train

        # initialize the first generation
        population = [self._random_series(n_gene_units, n_sample)
                 for i in range(self.groups)]

        # iterate to generate follow-up generations
        bests = []
        genes = []
        for i in range(self.iter):
            scores, population, gene = self._one_generation(population, adapt_func)
            bests.append(np.min(scores))
            self._verbose('Generation {0:3}: Best socre:{1}'.format(i, bests[-1]))
            genes.append(gene)

        self._verbose('Final best score:{0}'.format(bests[-1]))
        best_gene = genes[np.argmin(scores)]
        # return best_gene, the vary best scores
        return (best_gene, bests)
            
    
    def _one_generation(self, genes, adapt_func):
        scores = [adapt_func(gene) for gene in genes]
        best_gene = genes[np.argmin(scores)]

        board = self._gambling_board(scores)
        n_keep_best = int(self.groups * self.r_keep_best)
        # keep the bests
        bests_idx = np.array(np.argsort(scores)[:n_keep_best])
        #print(bests_idx)
        new_genes =(np.array(genes)[bests_idx, :]).tolist()
        # generate childrens
        pool = []
        board_sum = np.cumsum(board)
        while(len(pool) < len(genes)-len(new_genes)):
            n_childrens = self.groups - n_keep_best
            rands = random.randint(0, np.sum(board))
            chose = np.argmin(abs(board-rands))
            pool.append(genes[chose])
        while(len(new_genes) < len(genes)):
            rand1 = random.randint(0, len(pool)-1)
            rand2 = random.randint(0, len(pool)-1)
            if(adapt_func(pool[rand1]) > adapt_func(pool[rand2])):
                child1 = pool[rand1]
                child2 = pool[rand2]
            else:
                child1 = pool[rand2]
                child2 = pool[rand1]
            #new_genes.append(_reproduce(child1, child2))
            for j in range(np.shape(pool)[1]):
                rand3 = random.uniform(0, 1)
                if(rand3 < self.r_crossover):
                    child2[j] = child1[j]
            new_genes.append(child1)
            new_genes.append(child2)
        genes = new_genes
        return (scores, genes, best_gene)
