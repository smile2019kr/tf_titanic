import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from titanic.model import TitanicModel
class TitanicView:
    def __init__(self):
        self._m = TitanicModel()
        self._context = './data/'

    def create_train(self) -> object:
        m = self._m
        m.context = self._context
        m.fname = 'train.csv'
        t = m.new_dfame()
        return t

    @staticmethod
    def plot_survived_dead(train):
        f, ax = plt.subplots(1, 2, figsize=(18, 8))  # subplot 자동완성 오류
        # subplots로 의식적으로 코딩할 것
        train['Survived'].value_counts().plot.pie(explode=[0,0.1],
                                                  autopct='%1.1f%%',
                                                  ax=ax[0],
                                                  shadow=True)
        ax[0].set_title('Survived')
        ax[0].set_ylabel('')

        sns.countplot('Survived', data=train, ax=ax[1])
        ax[1].set_title('Survived')
        plt.show()

    @staticmethod
    def plot_sex(train): # 성별에 따른 생존자 비율 확인
        f, ax = plt.subplots(1, 2, figsize=(18, 8))  # subplot 자동완성 오류
        # subplots로 의식적으로 코딩할 것
        train['Survived'][trian['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],
                                                  autopct='%1.1f%%',
                                                  ax=ax[0],
                                                  shadow=True)
        train['Survived'][trian['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],
                                                  autopct='%1.1f%%',
                                                  ax=ax[1],
                                                  shadow=True)
        ax[0].set_title('Male')
        ax[1].set_title('Female')
#  ax[0].set_ylabel('')
#        sns.countplot('Survived', data=train, ax=ax[1])  # 전체중에서 생존자 비율을 확인하기 위한 코드
#        ax[1].set_title('Survived') # 전체중에서 생존자 비율을 확인하기 위한 코드
        plt.show()

    @staticmethod
    def bar_chart(train, feature):
        survived = train[train['Survived']==1][feature].value_counts()
        dead = train[train['Survived']==0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['survived', 'dead']
        df.plot(kind='bar', stacked=True, figsize = (110, 1))
        plt.show()
