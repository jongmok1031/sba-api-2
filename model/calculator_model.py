import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import tensorflow.compat.v1 as tf
from config import basedir
class CalculatorModel:
    def __init__(self):
        self.model = os.path.join(basedir, 'model')
        self.data = os.path.join(self.model, 'data')

    def create_add_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.add(w1, w2, name='op_add')
        sess = tf.Session()
        _ =  tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] :  {feed_dict['w1']}")
        print(f"feed_dict['w2'] :  {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 덧셈 결과 {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_add','model'), global_step=1000)

    def create_sub_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.subtract(w1, w2, name='op_sub')
        sess = tf.Session()
        _ =  tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] :  {feed_dict['w1']}")
        print(f"feed_dict['w2'] :  {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 뺄셈 결과 {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_sub','model'), global_step=1000)

    def create_mul_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.multiply(w1, w2, name='op_mul')
        sess = tf.Session()
        _ =  tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] :  {feed_dict['w1']}")
        print(f"feed_dict['w2'] :  {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 곱셈 결과 {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_mul','model'), global_step=1000)

    def create_div_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.math.floordiv(w1, w2, name='op_div')
        sess = tf.Session()
        _ =  tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] :  {feed_dict['w1']}")
        print(f"feed_dict['w2'] :  {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 나눗셈 결과 {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_div','model'), global_step=1000)

if __name__ == '__main__':
    print(f'* {basedir} *')
    c = CalculatorModel()
    c.create_div_model()