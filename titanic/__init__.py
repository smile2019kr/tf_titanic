from titanic.controller import TitanicController
from titanic.view import TitanicView
if __name__ == '__main__':
#    ctrl = TitanicController()
#    t = ctrl.create_train()

#2019.9.20 수업에서 추가된 내용
    def print_menu():
        print('0. EXIT')
        print('1. LEARNING MACHINE') #모델학습 후 정확도 측정하는데는 hook함수를 사용해야함
        print('2. VIEW : plot_survived_dead')
        print('3. TEST ACCURACY')
        print('4. SUBMIT')
        return input('CHOOSE ONE \n')

    while 1:
        menu = print_menu()
        print('MENU : %s' % menu)
        if menu == '0':
            print('** EXIT **')
            break
        elif menu == '1':
            print('** CREATE TRAIN **')
            ctrl = TitanicController()
            t = ctrl.create_train()
            print('** t 모델 **')
            print(t)
            break

        elif menu == '2':
            view = TitanicView()
            t = view.create_train()
          #  view.plot_survived_dead(t)  # 전체 중에서 생존자 비율 시각화 결과 확인하기 위한 코드
#            view.plot_sex(t) # 성별에 따른 생존자비율 시각화 결과 확인하기 위한 코드
            view.bar_chart(t, 'Pclass')
            break

        elif menu == '3':
            ctrl = TitanicController()
            t = ctrl.create_train()
            ctrl.test_all()

        elif menu == '4':
            ctrl = TitanicController()
            t = ctrl.create_train()
            ctrl.submit()
