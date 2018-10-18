import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
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
    '''for film in df.index:
        user = df.loc[film][~df.loc[film].isnull()]
        if len(user)!=0:
            i = random.randint(0, len(user)-1)
            user = user.index[i]
            note = copy.copy(df[user].loc[film])
            df[user].loc[film] = np.nan
            test.append((user,film,note))'''
    print('length test : ' + str(float(len(test))))
    print('ration test/train: ' + str(float(len(test))/float(info_tot)))
    return(df, test)

class Recommandation(object):

    def __init__(self, model):
        self.model = model

    def fit(self,database):
        self.database = database
        mtx = sp.csr_matrix(self.database.replace(np.nan, 0).values)
        self.model.fit(mtx)
        q = self.model.transform(mtx)
        self.q = pd.DataFrame(q, index=self.database.index)
        p = self.model.components_
        self.p = pd.DataFrame(p, columns=self.database.columns)

    def test(self, user, film):
        return (round((self.q.loc[film] * self.p[user]).sum(), 0))

    def score(self,list_test):
        list_error = []
        for (user,film,note) in list_test:
            note_test = self.test(user,film)
            list_error.append(abs(note-note_test))
        df_error = pd.DataFrame(list_error)
        return df_error.describe()