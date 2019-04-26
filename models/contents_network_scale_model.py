import tensorflow as tf
import numpy as np

def base_model(num_doc,num_user,real_batch_size,model_batch_data_idx):
    init_user_alpha = 1
    with tf.variable_scope("Ideal_Variable"):
        user_contents_alpha = tf.Variable(init_user_alpha, dtype=tf.float32)
        user_network_alpha = tf.Variable(init_user_alpha, dtype=tf.float32)

    contents_matrix = tf.mul(tf.ones(shape=[num_user, num_doc]), user_contents_alpha)
    network_matrix = tf.mul(tf.ones(shape=[num_user, num_doc]), user_network_alpha)

    contents_scale_matrix = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(contents_matrix), model_batch_data_idx)),
        [num_user, real_batch_size])
    network_scale_matrix = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(network_matrix), model_batch_data_idx)),
        [num_user, real_batch_size])

    return contents_scale_matrix,network_scale_matrix


def user_base_model(num_doc,num_user,real_batch_size,model_batch_data_idx,lambda_alpha):
    with tf.variable_scope("Ideal_Variable"):
        user_contents_alpha = tf.get_variable(name = "contents_alpha",
                                              initializer=tf.truncated_normal(shape=[1, num_user],mean=0,stddev=tf.truediv(1.0, lambda_alpha)))
        user_network_alpha = tf.get_variable(name = "network_alpha",
                                              initializer=tf.truncated_normal(shape=[1, num_user],mean=0,stddev=tf.truediv(1.0, lambda_alpha)))
    #user_contents_alpha = tf.Variable(init_user_alpha, dtype=tf.float32)
    #user_network_alpha = tf.Variable(init_user_alpha, dtype=tf.float32)

    contents_matrix = tf.transpose(tf.matmul(tf.ones(shape=[num_doc, 1]), user_contents_alpha))
    network_matrix = tf.transpose(tf.matmul(tf.ones(shape=[num_doc, 1]), user_network_alpha))

    contents_scale_matrix = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(contents_matrix), model_batch_data_idx)),
        [num_user, real_batch_size])
    network_scale_matrix = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(network_matrix), model_batch_data_idx)),
        [num_user, real_batch_size])

    return contents_scale_matrix,network_scale_matrix,user_contents_alpha,user_network_alpha

def doc_base_model(num_doc,num_user,real_batch_size,model_batch_data_idx):
    init_doc_alpha = np.ones([num_doc,1])

    with tf.variable_scope("Ideal_Variable"):
        doc_contents_alpha = tf.Variable(init_doc_alpha, dtype=tf.float32)
        doc_network_alpha = tf.Variable(init_doc_alpha, dtype=tf.float32)

    contents_matrix = tf.transpose(tf.matmul(doc_contents_alpha,tf.ones(shape=[1,num_user])))
    network_matrix = tf.transpose(tf.matmul(doc_network_alpha,tf.ones(shape=[1,num_user])))

    contents_scale_matrix = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(contents_matrix), model_batch_data_idx)),
        [num_user, real_batch_size])
    network_scale_matrix = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(network_matrix), model_batch_data_idx)),
        [num_user, real_batch_size])

    return contents_scale_matrix,network_scale_matrix


def user_doc_base_model(num_doc,num_user,real_batch_size,model_batch_data_idx):

    init_user_doc_alpha = np.ones([num_user,num_doc])
    with tf.variable_scope("Ideal_Variable"):
        user_doc_contents_alpha = tf.Variable(init_user_doc_alpha, dtype=tf.float32)
        user_doc_network_alpha = tf.Variable(init_user_doc_alpha, dtype=tf.float32)

    contents_scale_matrix = tf.reshape(tf.transpose(tf.gather(tf.transpose(user_doc_contents_alpha), model_batch_data_idx)),[num_user, real_batch_size])
    network_scale_matrix = tf.reshape(tf.transpose(tf.gather(tf.transpose(user_doc_network_alpha), model_batch_data_idx)),[num_user, real_batch_size])

    return contents_scale_matrix,network_scale_matrix

def without_network_model(num_doc,num_user,real_batch_size,model_batch_data_idx):
    contents_scale_matrix = tf.ones([num_user,num_doc])
    network_scale_matrix = tf.zeros([num_user,num_doc])

    contents_scale_matrix = tf.reshape(tf.transpose(tf.gather(tf.transpose(contents_scale_matrix), model_batch_data_idx)),[num_user, real_batch_size])
    network_scale_matrix = tf.reshape(tf.transpose(tf.gather(tf.transpose(network_scale_matrix), model_batch_data_idx)),[num_user, real_batch_size])

    return contents_scale_matrix,network_scale_matrix

def only_network_model(num_doc,num_user,real_batch_size,model_batch_data_idx):
    contents_scale_matrix = tf.zeros([num_user,num_doc])
    network_scale_matrix = tf.ones([num_user,num_doc])

    contents_scale_matrix = tf.reshape(tf.transpose(tf.gather(tf.transpose(contents_scale_matrix), model_batch_data_idx)),[num_user, real_batch_size])
    network_scale_matrix = tf.reshape(tf.transpose(tf.gather(tf.transpose(network_scale_matrix), model_batch_data_idx)),[num_user, real_batch_size])

    return contents_scale_matrix,network_scale_matrix