"""
Variable	Definition	Key
survival	생존여부	0 = No, 1 = Yes
pclass	승선권	1 = 1st, 2 = 2nd, 3 = 3rd
sex	성별
Age	나이
sibsp	동반한 형제, 자매, 배우자
parch	동반한 부모, 자식
ticket	티켓번호
fare	티켓요금
cabin	객실번호
embarked	승선한 항구명  C = 쉐브로, Q = 퀸즈타운, S = 사우스햄튼

데이터프레임에 들어가있는 변수명(이하의 Index 안에 으로 코딩에 활용해야 함(str이므로 변수명 인식에 대소문자 구분)
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics



class TitanicModel:
    def __init__(self):
        self._context = None
        self._fname = None
        self._train = None
        self._test = None
        self._test_id = None
#     def __init__(self, context, fname, train, .... ):
#     self.context = context
# 위와 같이 1-2회차 수업에서 했던 방식으로 코딩하는 것은 변수가 적을때는 가능하지만
# (instance 변수가 적을경우. 잘짜여진 학습도구가 만들어져있는경우)
# 변수의 개수가 늘어나면 매번 입력하기 어려우므로 None 으로 설정해주는 식으로 코딩
# feature 에 해당되는 변수명 지정시, self.--- 에 입력되는 변수명과 self.---에 들어가는 변수명을 구분할 필요
    # self.---입력시  _를 추가하는 식으로 구분(어떤식으로 다르게 표기할지는 공동작업자간의 협의)
    @property
    def context(self) -> object: return self._context
    #read only 로 설정. 괄호안에 context같은 변수명이 들어가있지않으므로 getter. context가 들어가야 하는 공간에 빈 괄호가 들어가있는 상태
    @context.setter
    def context(self, context): self._context = context
    # 판단하는 연산자가 들어가있으므로 setter
    # getter와 setter를 설정해주는 한쌍의 구문 (@property / @---.setter)

    @property
    def fname(self) -> object: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> object: return self._train

    @train.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test

    @test.setter
    def test(self, test): self._test = test

    @property
    def test_id(self) -> object: return self._test_id

    @test_id.setter
    def test_id(self, test_id): self._test_id = test_id

    # 람다학습?( AWS Lambda )을 활용하기 위한 구문설정
    # 서버를 따로 구축하지않아도 (자체 서버 구축 안해도) 메모리할당없이 활용가능

    def new_file(self) -> str: return self._context + self._fname

    def new_dfame(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

#hook 메서드 (구글링으로 상세내용 확인)
    def hook_process(self, train, test) -> object:
        #이하의 필터링으로 feature값을 줄여나가면서 accuracy높이는 효율적인 모형 산출
        print('----------------1. Cabin Ticket 삭제 --------------------------')
        t = self.drop_feature(train, test, 'Cabin')
        t = self.drop_feature(t[0], t[1], 'Ticket')
        print('----------------2. embarked 승선한 항구명 norminal 편집 --------------------------')
        t = self.embarked_nominal(t[0], t[1])
        print('----------------3. Title 편집--------------------------')
        t = self.title_norminal(t[0], t[1])
        print('----------------4. Name, PassengerId 삭제-------------------------')
        t = self.drop_feature(t[0], t[1], 'Name')
        self._test_id = test['PassengerId']
        # test에서 활용하기 위해 PassengerID를 test_id에 저장해두고 train-test용 데이터에서는 삭제함
        t = self.drop_feature(t[0], t[1], 'PassengerId')
        print('----------------5. Age ordinal 편집--------------------------')
        t = self.age_ordinal(t[0], t[1])
        print('----------------6. Fare ordinal 편집--------------------------')
        t = self.fare_ordinal(t[0], t[1])
        print('----------------7. Fare 삭제 -------------------------')
        t = self.drop_feature(t[0], t[1], 'Fare')
        print('----------------8. Sex norminal 편집 -------------------------')
        t = self.sex_norminal(t[0], t[1])
        t[1] = t[1].fillna({"FareBand": 1})
        # 결손치 (공백값) 체크 구문
        a = self.null_sum(t[1])
        print('null 수량 {} 개'.format(a))
        self._test = t[1]
        return t[0]

    @staticmethod
    def null_sum(train) -> int:
        return train.isnull().sum()
     #   sum = train.isnull().sum()
     #   return sum


    @staticmethod
    def drop_feature(train, test, feature) -> []:
        train = train.drop([feature], axis = 1)
        test = test.drop([feature], axis = 1)
        return [train, test]

    @staticmethod
        # train, test값을 항상 함께 제시해야하므로 결과값을 list형태[]로 지정
    def embarked_nominal(train, test) -> []:
     #   c_city = train[train['Embarked'] == 'C'].shape[0]
     #   s_city = train[train['Embarked'] == 'S'].shape[0]
     #   q_city = train[train['Embarked'] == 'Q'].shape[0]
     # 한줄씩 뽑아내는 형태의 코딩

        train = train.fillna({"Embarked" : "S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        train['Embarked'] = train['Embarked'].map(city_mapping)
        test['Embarked'] = test['Embarked'].map(city_mapping)

        print('----------- train head & column -----------------')
        print(train.head())
        print(train.columns)
        return [train, test]

    @staticmethod
    def title_norminal(train, test) -> []:
        combine = [train, test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
            # 이름을 삭제하고 신분 Mrs. 등에 해당되는 글자만 추출. '[A-Za-z]+\.' 으로 할 경우 A-Za-z 한글자에 해당하는것만 추출

        for dataset in combine:
            dataset['Title'] \
                = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
            # 데이터셋에서 귀족 등의 고급계층을 드러내는 단어들을 추출하여 'Rare'라는 단어로 대체
            dataset['Title'] \
                = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

            dataset['Title'] \
                = dataset['Title'].replace(['Mile', 'Ms'], 'Miss')

            dataset['Title'] \
                = dataset['Title'].replace(['Mne', 'Mrs'], 'Mrs')

        train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
        # 계층이라는 feature가 생존여부에 어느정도로 연관이 있는지를 검토하기 위한 결과출력

        title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6, 'Mne':7}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        return [train, test]


    @staticmethod
    def sex_norminal(train, test) -> []:
        combine = [train, test]
        sex_mapping = {'male':0, 'female':1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)

        return [train, test]

    @staticmethod
    def age_ordinal(train, test) -> []:
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        # -0.5로 지정한 이유는 이하의 -1과 0의 사이구간에 unknown값을 할당하기 위함
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        # 구간 지정을 위한 구문
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)

        age_title_mapping = {0: 'Unknown', 1: 'Baby', 2: 'Child', 3: 'Teenager',
                             4: 'Student', 5: 'Young Adult', 6: 'Adult', 7: 'Senior'}
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]

        age_mapping = {'Unknown':0, 'Baby':1, 'Child':2, 'Teenager':3,
                             'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}

        train['AgeGroup'] = train['AgeGroup'] .map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        print(train['AgeGroup'].head())
        return [train, test]

    @staticmethod
    def fare_ordinal(train, test) -> []:
        train['FareBand'] = pd.qcut(train['Fare'], 4, labels={1,2,3,4})
        test['FareBand'] = pd.qcut(test['Fare'], 4, labels={1, 2, 3, 4})
        # 엄밀하게 구분하지 않아도 될 경우 qcut을 사용하면 4군으로 분류
        return [train, test]

    #검증 알고리즘 작성 (2019.9.20 수업에서 추가)

    def hook_test(self, model, dummy):
        print('KNN 활용한 검증 정확도 {} %'.format(self.accuracy_by_knn(model, dummy)))
        print('결정트리 활용한 검증 정확도 {} %'.format(self.accuracy_by_dtree(model, dummy)))
        print('랜덤포레스트 활용한 검증 정확도 {} %'.format(self.accuracy_by_rforest(model, dummy)))
        print('나이브베이즈 활용한 검증 정확도 {} %'.format(self.accuracy_by_nb(model, dummy)))
        print('SVM 활용한 검증 정확도 {} %'.format(self.accuracy_by_svm(model, dummy)))

    @staticmethod
    def create_k_fold():
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        return k_fold
    # KFold와 randomvariables로 데이터 더미를 잔뜩 만들어두기 위한 코딩
    @staticmethod
    def create_random_variables(train, X_feature, Y_features) -> []:
        the_X_feature = X_feature
        the_Y_feature = Y_features
        train2, test2 = train_test_split(train, test_size=0.3, random_state=0)
        train_X = train2[the_X_feature]
        train_Y = train2[the_Y_feature]
        test_X = test2[the_X_feature]
        test_Y = test2[the_Y_feature]
        return [train_X, train_Y, test_X, test_Y]

    def accuracy_by_knn(self, model, dummy):
        clf = KNeighborsClassifier(n_neighbors=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2) # 소수점 2번째자리에서 자름
        return accuracy

    def accuracy_by_dtree(self, model, dummy):
        clf = DecisionTreeClassifier()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)  # 소수점 2번째자리에서 자름
        return accuracy

    def accuracy_by_rforest(self, model, dummy):
        clf = RandomForestClassifier(n_estimators=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)  # 소수점 2번째자리에서 자름
        return accuracy

    def accuracy_by_nb(self, model, dummy):
        clf = GaussianNB()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)  # 소수점 2번째자리에서 자름
        return accuracy

    def accuracy_by_svm(self, model, dummy):
        clf = SVC()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)  # 소수점 2번째자리에서 자름
        return accuracy

