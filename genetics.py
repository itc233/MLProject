import numpy as np
import random
from multiprocessing import Pool

class GA(object):

    # the last column of train and valid will be taken as the perdict value
    def __init__(self, train, valid, estimator, feval, 
                 iter=200, r_sample=0.6, r_crossover=0.5, r_vary=0.01,
                 r_keep_best=0.1, popsize = 1000,
                 verbose=False):
        self.train = train
        self.origin_estimator = estimator
        self.estimator = estimator
        self.verbose = verbose
        self.iter = iter
        self.r_sample = r_sample
        self.valid = valid
        self.r_crossover = r_crossover
        self.r_vary = r_vary
        self.r_keep_best = r_keep_best
        self.popsize = popsize
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
    def select(self):
        return self._selectFeature()

    # the smaller the better
    def _selectFeatureScore(self, gene):
        self.estimator = self.origin_estimator
        sample = self.train.T[:-1][gene].T
        estor = self.estimator.fit(sample, self.train.T[-1].T)
        valid_fs = self.valid.T[:-1][gene].T
        predicts = estor.predict(valid_fs)
        return self.feval(predicts, self.valid.T[-1].T)

    def _selectFeature(self):
        n_features = self.train.shape[1] - 1  # the last feature is the value
        n_sample = int(self.r_sample * n_features)

        (best_gene, best_scores) = self._run(
            n_features, n_sample, self._selectFeatureScore
        )

        best_sample = self.train.T[:-1][best_gene].T
        return (best_sample, best_gene, best_scores)

    # generate a random array whose len=n_len, and have n_pos's value==True
    def _randomSeries(self, n_len, n_pos):
        gene = np.zeros(n_len, dtype=np.bool)

        indexes = np.arange(n_len, dtype=np.int)
        random.shuffle(indexes)

        gene[indexes[:n_pos]] = True
        return gene

    def _gamblingBoard(self, scores):
        # scores: the smaller the better
        # possbilities: the bigger the better
        # p = (alpha)^score

        alpha = 0.8
        tmp = list(map(lambda x: pow(alpha, x), scores))
        tmp = tmp / np.sum(tmp)               # calculate p
        possbilities = tmp #/ np.min(tmp)      # scale
        return possbilities
    
    def _run(self, n_gene_units, n_sample, adapt_func):
        if n_gene_units == n_sample:
            return self.train
        population = [self._randomSeries(n_gene_units, n_sample)
                 for i in range(self.popsize)]
        bests = []
        genes = []
        for i in range(self.iter):
            scores, population, gene = self._oneGeneration(population, adapt_func)
            bests.append(np.min(scores))
            m_genes = np.mean(population, axis = 0)
            #print(m_genes)
            self._verbose('Generation {0:3}: Best socre:{1}'.format(i, bests[-1]))
            genes.append(gene)
            np.save('trained_genes', gene)
            print("in main ", np.min(scores))
        self._verbose('Final best score:{0}'.format(bests[-1]))
        best_gene = genes[np.argmin(scores)]
        # return best_gene, the vary best scores
        return (best_gene, bests)
            
    def _getChildId(self, cumboard, sort_id):
        val = random.uniform(0, cumboard[len(cumboard)-1])
        ans = 0
        for id, i in enumerate(cumboard):
            if val <= i:
                ans = sort_id[id]
                break
        return ans

    def _swapGene(self, ch1, ch2):
        return ch2, ch1

    def _getChild(self, cumboard, sort_id, scores, genes):
        idx = self._getChildId(cumboard, sort_id)
        return genes[idx], scores[idx]

    def _oneGeneration(self, genes, adapt_func):
        scores = [adapt_func(gene) for gene in genes]
        board = self._gamblingBoard(scores)

        best_gene = genes[np.argmin(scores)]
        n_keep_best = int(len(genes) * self.r_keep_best)
        bests_idx = np.array(np.argsort(scores)[:n_keep_best])
        print("best score", np.array(scores)[bests_idx])
        new_genes =(np.array(genes)[bests_idx, :]).tolist()
        print("best gene", best_gene)

        new_best = new_genes[0]
        print("best new gene", new_best)
        print("before pooling ", adapt_func(new_best))
        sort_id = np.argsort(board)
        cumboard = np.cumsum(np.sort(board))

        while(len(new_genes) < len(genes)):
            ch1, sc1 = self._getChild(cumboard, sort_id, scores, genes)
            ch2, sc2 = self._getChild(cumboard, sort_id, scores, genes)
            if(sc1 > sc2):
                ch1, ch2 = self._swapGene(ch1, ch2)
            for j in range(len(ch1)):
                do_cross = random.uniform(0, 1)
                if(do_cross < self.r_crossover):
                    ch2[j] = ch1[j]
                do_mut = random.uniform(0, 1)
                if(do_mut < self.r_vary):
                    ch1[j] = !ch1[j]
                do_mut = random.uniform(0, 1)
                if(do_mut < self.r_vary):
                    if(ch2[j]):
                        ch2[j] = False
                    else :
                        ch2[j] = True
            new_genes.append(ch1)
            new_genes.append(ch2)

        #genes = (np.array(new_genes)[len(genes), :]).tolist()
        new_best = np.array(new_genes)[0, :]
        print("after pooling ", adapt_func(new_best))
        return (scores, new_genes, best_gene)
