import tensorflow as tf
from keras import backend as K

def arc_face(labels, predictions):
    session = K.get_session()
    weights = sess.graph.get_tensor_by_name("dense_3/kernel")
    print(weights)
