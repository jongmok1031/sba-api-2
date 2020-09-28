from config import basedir
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import tensorflow.compat.v1 as tf


class CalculatorService:
   
    def __init__(self):
        self.model = os.path.join(basedir, 'model')

    def calc(self, num1, num2, opcode):
        print(f'서비스에 전달된 num1 : {num1}, num2 : {num2}, opcode: {opcode}')
        tf.reset_defult_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            saver = tf.train.import_meta_graph(self.model + '/calculator_'+opcode + '/model-1000.meta' )

            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name('w1:0')
            w2 = graph.get_tensor_by_name('w2:0')
            feed_dict = {w1: float(num1), w2: float(num2)}
            op_to_restore = graph.get_tensor_by_name('op_'+opcode+':0')
            result = sess.run(op_to_restore, feed_dict)
            print(f'최종결과: {result}')

        return result