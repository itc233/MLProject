import numpy as np
import random
from multiprocessing import Pool

class GA(object):

    # the last column of train and valid will be taken as the perdict value
    def __init__(self, Xdata, estimator, feval, 
                 iter=200, r_sample=0.6, r_crossover=0.5, r_vary=0.01,
                 r_keep_best=0.1, popsize = 1000, pTrain = 0.8,
                 verbose=False):
        self.Xdata = Xdata
        self.origin_estimator = estimator
        self.estimator = estimator
        self.verbose = verbose
        self.iter = iter
        self.r_sample = r_sample
        self.r_crossover = r_crossover
        self.r_vary = r_vary
        self.r_keep_best = r_keep_best
        self.popsize = popsize
        self.feval = feval
        self.pTrain = pTrain
        self.train = []
        self.valid = []
        self._generateData()
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
        Xtr = self.train[:, :-1]
        Xtr = Xtr[:, gene]
        ytr = self.train[:, -1]

        Xvalid = self.valid[:, :-1]
        Xvalid = Xvalid[:, gene]
        yvalid = self.valid[:, -1]

        self.estimator = self.origin_estimator
        estor = self.estimator.fit(Xtr, ytr)
        predicts = estor.predict(Xvalid)
        return self.feval(predicts, yvalid)

    def _selectFeature(self):
        n_features = self.train.shape[1] - 1  # the last feature is the value
        n_sample = int(self.r_sample * n_features)

        (best_gene, best_scores) = self._run( n_features, n_sample, self._selectFeatureScore )

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
        possbilities = tmp / np.min(tmp)      # scale
        return possbilities

    def _generateDataIdx(self):
        tr_len = np.int(self.pTrain*len(self.Xdata))
        
        rand_idx = np.arange(len(self.Xdata))
        np.random.shuffle(rand_idx)
        tr_idx = rand_idx[:tr_len]
        vald_idx = rand_idx[tr_len:]
        return tr_idx, vald_idx

    def _generateData(self):
        tr_idx, val_idx = self._generateDataIdx()
        self.train = self.Xdata[tr_idx, :]
        self.valid = self.Xdata[val_idx, :]
        return

    def _run(self, n_gene_units, n_sample, adapt_func):
        if n_gene_units == n_sample:
            return self.train
        population = [self._randomSeries(n_gene_units, n_sample)
                 for i in range(self.popsize)]
        scoresHist = []
        genesHist = []
        for i in range(self.iter):
            self._generateData()
            scores, population, gene = self._oneGeneration(population, adapt_func)
            scoresHist.append(np.min(scores))
            genesHist.append(gene)

            m_genes = np.mean(population, axis = 0)
            print(m_genes)

            self._verbose('Generation {0:3}: Best score this gen:{1} Best socre:{1}'.format(i, np.min(scores) ,scoresHist[-1]))

            np.save('trained_genes', gene)
            
        self._verbose('Final best score:{0}'.format(scoresHist[-1]))
        best_gene = genesHist[-1]
        return (best_gene, scoresHist)
            
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

    def _getChild(self, scores, genes, board):
        cumboard = np.cumsum(np.sort(board))
        sort_id = np.argsort(board)
        idx = self._getChildId(cumboard, sort_id)
        board[idx] = board[idx]-1
        return genes[idx], scores[idx], board

    def _oneGeneration(self, genes, adapt_func):
        scores = [adapt_func(gene) for gene in genes]
        board = self._gamblingBoard(scores)

        best_gene = genes[np.argmin(scores)]
        n_keep_best = int(len(genes) * self.r_keep_best)
        bests_idx = np.array(np.argsort(scores)[:n_keep_best])
        new_genes =(np.array(genes)[bests_idx, :]).tolist()


        while(len(new_genes) < len(genes)):
            ch1, sc1, board = self._getChild(scores, genes, board)
            ch2, sc2, board = self._getChild(scores, genes, board)
            if(sc1 > sc2):
                ch1, ch2 = self._swapGene(ch1, ch2)
            for j in range(len(ch1)):
                do_cross = random.uniform(0, 1)
                if(do_cross < self.r_crossover):
                    ch2[j] = ch1[j]
                do_mut = random.uniform(0, 1)
                if(do_mut < self.r_vary):
                    ch1[j] = not ch1[j]
                do_mut = random.uniform(0, 1)
                if(do_mut < self.r_vary):
                    ch2[j] = not ch2[j]
            new_genes.append(ch1)
            new_genes.append(ch2)

        return (scores, new_genes, best_gene)
