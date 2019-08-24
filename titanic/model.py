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

데이터프레임에 들어가있는 변수명으로 코딩에 활용해야 함(str이므로 변수명 인식에 대소문자 구분)
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

"""

import pandas as pd
import numpy as np


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
        return t[0]

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

