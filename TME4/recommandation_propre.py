import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
import copy
import random





def cut_train_test(df):
    test = []
    info_tot = df.count().sum()
    for user in df.columns:
        film = df[user].loc[~df[user].isnull()]
        if len(film) != 0:
            i = random.randint(0,len(film)-1)
            film = film.index[i]
            note = copy.copy(df[user].loc[film])
            df[user].loc[film] = np.nan
            test.append((user,film,note))
    for film in df.index:
        user = df.loc[film][~df.loc[film].isnull()]
        if len(user)!=0:
            i = random.randint(0, len(user)-1)
            user = user.index[i]
            note = copy.copy(df[user].loc[film])
            df[user].loc[film] = np.nan
            test.append((user,film,note))
    print('length test : ' + str(float(len(test))))
    print('ration test/train: ' + str(float(len(test))/float(info_tot)))
    return(df, test)

def abs_error(y, ypred):
    return abs(y - ypred)

def rel_error(y, ypred):
    return (abs(y - ypred) / y)

class Recommandation(object):

    def __init__(self, model = None):
        self.model = model

    def fit(self,database):
        self.database = database
        mtx = sp.csr_matrix(self.database.replace(np.nan, 3.6).values)
        self.model.fit(mtx)
        q = self.model.transform(mtx)
        self.q = pd.DataFrame(q, index=self.database.index)
        p = self.model.components_
        self.p = pd.DataFrame(p, columns=self.database.columns)

    def fit_SGD_stoch(self, database, dim, epsilon, reg_q, reg_p, max_iter,test):
        self.database = database
        Y = torch.tensor(np.array(database.replace(np.nan, 3.6)))
        q = torch.tensor(np.random.rand(Y.size(0), dim))
        p = torch.tensor(np.random.rand(dim, Y.size(1)))
        cost = []
        for it in range(0, max_iter):
            ypred = torch.mm(q, p)
            delta = Y - ypred
            cost.append(torch.mul(delta,delta).sum())
            q_n = (1 - epsilon * reg_q) * q + epsilon * torch.mm(p, delta.t()).t()
            p_n = (1 - epsilon * reg_p) * p + epsilon * torch.mm(delta.t(), q).t()
            q = q_n
            p = p_n
        self.q = pd.DataFrame(np.array(q), index=self.database.index)
        self.p = pd.DataFrame(np.array(p), columns=self.database.columns)

        return(cost)

    def test(self, user, film):
        note = round((self.q.loc[film] * self.p[user]).sum(), 0)
        return (note)

    def score(self,list_test,error):
        list_error = []
        for (user,film,note) in list_test:
            note_test = self.test(user,film)
            list_error.append(error(note,note_test))
        df_error = pd.DataFrame(list_error)
        return df_error.describe()

    def random_score(self,list_test, error):
        list_error = []
        for (user, film, note) in list_test:
            note_test = random.randint(1,6)
            list_error.append(error(note, note_test))
        df_error = pd.DataFrame(list_error)
        return df_error.describe()

    def cst_score(self,list_test,note_cst, error):
        list_error = []
        for (user, film, note) in list_test:
            note_test = note_cst
            list_error.append(error(note,note_test))
        df_error = pd.DataFrame(list_error)
        return df_error.describe()

