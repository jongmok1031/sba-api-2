import sys
sys.path.insert(0,'/Users/jongm/SBAprojects')
from util.file_handler import FileReader
import pandas as pd
import numpy as np
import tensorflow as tf

class Cabbage:

    # year,  avgTemp,  minTemp,maxTemp,rainFall,avgPrice
    # 20100101,-4.9,   -11,      0.9,     0,      2123
    # 20100102,-3.1,   -5.5,     5.5,    0.8,    2123
    
    # 멤버 변수
    year: int = 0
    avgTemp: float = 0.0
    minTemp: float = 0.0
    maxTemp: float = 0.0
    rainFall: float = 0.0
    avgPrice: int = 0

    # 클래스 내부에서 공유하는 객체
    def __init__(self):
        self.fileReader = FileReader()   #기능은 상수처리
        self.context = '/Users/jongm/SBAprojects/price_prediction/data/'

    def new_model(self, payload) :
        this = self.fileReader
        this.context = self.context
        this.fname = payload
        return pd.read_csv( this.context + this.fname, sep=',')

    def create_tf(self, payload):
        xy = np.array(payload, dtype=np.float32)
        x_data = xy[:,1:-1] # feature
        y_data = xy[:,[-1]] # price # label
        X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
        Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random.normal([4, 1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')
        hyposthesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hyposthesis - Y))
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hyposthesis, train],
                                        feed_dict={X: x_data, Y: y_data})
            if step % 1000 == 0:
                print(f'# {step} 손실비용: {cost_} ')
                print(f'- 배추가격 : {hypo_[0]}')

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, self.context + 'saved_model.ckpt')
        print('저장완료')

    def test(self):
        self.avgPrice = 100
        print(self.avgPrice)

    def service(self):
        X = tf.compat.v1.placeholder(tf.float32, shape = [None,4])
        # year avgtemp mintemp maxtemp rainfall avgprice
        # 에서 x 변수 feature만 입력받겟다
        # year 불필요한 데이터
        # avgprice 는 label 종복
        # 종속변수 label을 결정하는 독립변수 feature
        # avgprice를 결정하는 요소로 사용되는 파라미터 (중요!!)
        # 통계 확률로 들어가자
        # 외부에서 주입되는 값 파라미터
        # y = ax+ b linear relationship
        # X 는 대문자 사용, 확률변수
        # 비교, 웹프로그래밍(java, c) 소문자 x이렇게 하는데 이건 한타임에 하나의 value
        # 그리고 그 값은 외부에서 주어지는 하나의 값이므로 그냥 변수
        # 지금은 X 의 값이 제한적이지만 집합상태로 많은값이 있는 상태
        # 이럴떄는 확률변수. 
        # 대문자 확률 소문자 하나로결정된 value
        W = tf.Variable(tf.random.normal([4,1]), name = 'weight')
        b = tf.Variable(tf.random.normal([1]), name = 'bias')
        # 텐서에서의 변수는 웹에서 변수화 다름
        # 이 변수를 결정하는건 외부값이 아닌 텐서가 내부에서 사용하는 변수
        # 기존웹에서 사용하는 변수는 placeholder

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, self.context+'saved_model.ckpt')
            print(f'avgTemp :{self.avgTemp} , minTemp: {self.minTemp}, maxTemp: {self.maxTemp}, rainFall: {self.rainFall}')
            data = [[self.avgTemp, self.minTemp, self.maxTemp, self.rainFall],]
            arr = np.array(data, dtype = np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            # Y = WX + b 를 코드로 표현하면 위 처럼
            # y = wx + b
            print(dict[0])
        return int(dict[0])


if __name__ == '__main__':
    cabbage = Cabbage()
    dframe = cabbage.new_model('price_data.csv')
    print(dframe.head())
    cabbage.create_tf(dframe)
    print(cabbage.avgPrice)