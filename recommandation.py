
import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

class Recommandation(object):

    def __init__(self,database):
        self.database = database

    def calcul_dist(self,df,key):
        '''Calcul la distance a un objet des elements de la base'''
        pass

    def naive_rec(self,user,film):
        df = self.database
        dist_film = self.calcul_dist(df, film)
        dist_user = self.calcul_dist(df.T, user)
        sim_users = dist_user.index[1:10]
        tot = 0.0
        note = 0.0
        for sim_us in sim_users:
            grades = pd.concat([df[sim_us].reindex(dist_film.index), dist_film], axis=1)
            col = grades.columns[0]
            rec_note = grades[grades[col] > 0.0].ix[0, col]
            tot += dist_user.loc[sim_us]
            note += dist_user.loc[sim_us] * rec_note
        return(note / tot)

    def prod_vec(self,user,film):
        df = self.database
        dist_film, dist_user = self.red_dim(df)
        return((dist_film[film]*dist_user[user]).sum())


class Recommandation_SVD(Recommandation):

    def red_dim(self,df):
        '''Calcul la distance a un objet des elements de la base'''
        mtx = sp.csr_matrix(df.replace(np.nan, 0).values)
        svd = TruncatedSVD(n_components=2)
        svd.fit(mtx)
        svd.fit(mtx)
        red_mtx_2 = svd.transform(mtx)
        return(pd.DataFrame(svd.inverse_transform(red_mtx_2)))

    def calcul_dist(self,df,key):
        '''Calcul la distance a un objet des elements de la base'''
        mtx = sp.csr_matrix(df.replace(np.nan,0).values)
        svd = TruncatedSVD(n_components=2)
        svd.fit(mtx)
        red_mtx = svd.fit_transform(mtx)

        df_film_red =  pd.DataFrame(red_mtx, index=df.index)
        df_dist_film = pd.DataFrame([0.0] * len(df), index=df.index)
        film_rec = key
        vec_rec = df_film_red.loc[film_rec]

        for row in df_film_red.index:
            vec = df_film_red.loc[row]
            dist = ((vec - vec_rec) * (vec - vec_rec)).sum()
            df_dist_film.loc[row][0] = dist

        return(df_dist_film.sort_values([0]))


class Recommandation_NMF(Recommandation):

    def calcul_dist(self,df,key):
        '''Calcul la distance a un objet des elements de la base'''
        mtx = sp.csr_matrix(df.replace(np.nan,0).values)
        svd = NMF(n_components=2, init='random', random_state=0)
        svd.fit(mtx)
        red_mtx = svd.fit_transform(mtx)

        df_film_red = pd.DataFrame(red_mtx, index=df.index)
        df_dist_film = pd.DataFrame([0.0] * len(df), index=df.index)
        film_rec = key
        vec_rec = df_film_red.loc[film_rec]

        for row in df_film_red.index:
            vec = df_film_red.loc[row]
            dist = ((vec - vec_rec) * (vec - vec_rec)).sum()
            df_dist_film.loc[row][0] = dist

        return(df_dist_film.sort_values([0]))
