import numpy as np
import os
from numpy import inf
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm
from functional import compose, partial
import functools

def evaluation(test_R,test_mask_R,Estimated_R,num_test_ratings):
    test_R = np.multiply((test_R+1)*0.5 , test_mask_R)
    pre_numerator = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.square(pre_numerator))
    denominator = num_test_ratings
    RMSE = np.sqrt(numerator / float(denominator))

    pre_numeartor = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.abs(pre_numeartor))
    denominator = num_test_ratings
    MAE = numerator / float(denominator)

    pre_numeartor1 = np.sign(Estimated_R - 0.5)
    tmp_test_R = np.sign(test_R - 0.5)

    pre_numerator2 = np.multiply((pre_numeartor1 == tmp_test_R), test_mask_R)
    numerator = np.sum(pre_numerator2)
    denominator = num_test_ratings
    ACC = numerator / float(denominator)

    a = np.log(Estimated_R)
    b = np.log(1 - Estimated_R)
    a[a == -inf] = 0
    b[b == -inf] = 0

    tmp_r = test_R
    tmp_r = a * (tmp_r > 0) + b * (tmp_r == 0)
    tmp_r = np.multiply(tmp_r, test_mask_R)
    numerator = np.sum(tmp_r)
    denominator = num_test_ratings
    AVG_loglikelihood = numerator / float(denominator)

    return RMSE,MAE,ACC,AVG_loglikelihood

def evaluation_not_voting(test_R,test_mask_R,overall_prob,
                          num_test_ratings,num_aye_ratings,num_nay_ratings,num_notvoting_ratings):
    # overall_prob : (self.prob_nay, self.prob_notvote, self.prob_aye)
    overall_denominator = num_test_ratings
    aye_denominator = num_aye_ratings
    nay_denominator = num_nay_ratings
    notvoting_denominator = num_notvoting_ratings

    max_idx = overall_prob.argmax(2) - 1
    overall_pre_numerator = (test_R == max_idx)
    vote_aye = (test_R == 1)
    vote_nay = (test_R == -1)
    vote_notvoting = (test_R == 0)

    ''' ACC '''
    aye_pre_numerator = overall_pre_numerator & vote_aye
    nay_pre_numerator = overall_pre_numerator & vote_nay
    notvoting_pre_numerator = overall_pre_numerator & vote_notvoting

    overall_numerator = np.sum(np.multiply(overall_pre_numerator, test_mask_R))
    aye_numerator = np.sum(np.multiply(aye_pre_numerator, test_mask_R))
    nay_numerator = np.sum(np.multiply(nay_pre_numerator, test_mask_R))
    notvoting_numerator = np.sum(np.multiply(notvoting_pre_numerator, test_mask_R))

    overall_ACC = overall_numerator / float(overall_denominator)
    aye_ACC = aye_numerator / float(aye_denominator)
    nay_ACC = nay_numerator / float(nay_denominator)
    notvoting_ACC = notvoting_numerator / float(notvoting_denominator)

    ''' RMSE '''
    aye_bool_numerator = ((test_R == 1) & (test_mask_R == 1))
    aye_pre_numerator = np.multiply((test_R - aye_pre_numerator), aye_bool_numerator)
    aye_numerator = np.sum(np.square(aye_pre_numerator))
    aye_RMSE = np.sqrt(aye_numerator / float(aye_denominator))

    nay_bool_numerator = ((test_R == -1) & (test_mask_R == 1))
    nay_pre_numerator = np.multiply((test_R - nay_pre_numerator), nay_bool_numerator)
    nay_numerator = np.sum(np.square(nay_pre_numerator))
    nay_RMSE = np.sqrt(nay_numerator / float(nay_denominator))

    notvoting_bool_numerator = ((test_R == 0) & (test_mask_R == 1))
    notvoting_pre_numerator = np.multiply((test_R - notvoting_pre_numerator), notvoting_bool_numerator)
    notvoting_numerator = np.sum(np.square(notvoting_pre_numerator))
    notvoting_RMSE = np.sqrt(notvoting_numerator / float(notvoting_denominator))

    overall_RMSE = np.sqrt((aye_numerator + nay_numerator + notvoting_numerator) / float(num_test_ratings))

    ''' MAE '''
    aye_numerator = np.sum(np.abs(aye_pre_numerator))
    aye_MAE = aye_numerator / float(aye_denominator)

    nay_numerator = np.sum(np.abs(nay_pre_numerator))
    nay_MAE = nay_numerator / float(nay_denominator)

    notvoting_numerator = np.sum(np.abs(notvoting_pre_numerator))
    notvoting_MAE = notvoting_numerator / float(notvoting_denominator)

    overall_numerator = np.sum(aye_numerator + nay_numerator + notvoting_numerator)
    overall_MAE = overall_numerator / float(num_test_ratings)

    ''' log likelihood '''
    # overall_prob : (self.prob_nay, self.prob_notvote, self.prob_aye)
    aye_avg = 0.5 * np.multiply(np.multiply(test_R , 1 + test_R) , np.log(overall_prob[:,:,2] + 1e-10))
    notvoting_avg = np.multiply(np.multiply(1 + test_R, 1 - test_R) , np.log(overall_prob[:,:,1] + 1e-10))
    nay_avg = 0.5 * np.multiply(np.multiply(test_R , test_R - 1) , np.log(overall_prob[:,:,0] + 1e-10))
    AVG_loglikelihood = np.sum(np.multiply(aye_avg + notvoting_avg + nay_avg , test_mask_R)) / float(num_test_ratings)

    return overall_ACC,aye_ACC,nay_ACC,notvoting_ACC,\
           overall_RMSE,aye_RMSE,nay_RMSE,notvoting_RMSE,\
           overall_MAE,aye_MAE,nay_MAE,notvoting_MAE,\
           AVG_loglikelihood

def make_records_original(result_path,test_acc_list,test_rmse_list,test_mae_list,test_avg_loglike_list,current_time,
                 args,model_name,data_name,train_ratio,hidden_neuron,random_seed,optimizer_method,lr):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    overview = 'results/' + 'overview.txt'
    basic_info = result_path + "basic_info.txt"
    test_record = result_path + "test_record.txt"

    with open(test_record, 'w') as g:

        g.write(str("ACC:"))
        g.write('\t')
        for itr in range(len(test_acc_list)):
            g.write(str(test_acc_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RMSE:"))
        g.write('\t')
        for itr in range(len(test_rmse_list)):
            g.write(str(test_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("MAE:"))
        g.write('\t')
        for itr in range(len(test_mae_list)):
            g.write(str(test_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("AVG Likelihood:"))
        g.write('\t')
        for itr in range(len(test_avg_loglike_list)):
            g.write(str(test_avg_loglike_list[itr]))
            g.write('\t')
        g.write('\n')

    with open(basic_info, 'w') as h:
        h.write(str(args))

    with open(overview, 'a') as f:
        f.write(str(data_name))
        f.write('\t')
        f.write(str(model_name))
        f.write('\t')
        f.write(str(train_ratio))
        f.write('\t')
        f.write(str(current_time))
        f.write('\t')
        f.write(str(test_rmse_list[-1]))
        f.write('\t')
        f.write(str(test_mae_list[-1]))
        f.write('\t')
        f.write(str(test_acc_list[-1]))
        f.write('\t')
        f.write(str(test_avg_loglike_list[-1]))
        f.write('\t')
        f.write(str(hidden_neuron))
        f.write('\t')
        f.write(str(args.corruption_level))
        f.write('\t')
        f.write(str(args.lambda_value))
        f.write('\t')
        f.write(str(args.lambda_t_value))
        f.write('\t')
        f.write(str(args.lambda_y))
        f.write('\t')
        f.write(str(args.lambda_u))
        f.write('\t')
        f.write(str(args.lambda_w))
        f.write('\t')
        f.write(str(args.lambda_n))
        f.write('\t')
        f.write(str(args.lambda_tau))
        f.write('\t')
        f.write(str(args.lambda_f))
        f.write('\t')
        f.write(str(args.lambda_alpha))
        f.write('\t')
        f.write(str(args.f_act))
        f.write('\t')
        f.write(str(args.g_act))
        f.write('\n')

    Test = plt.plot(test_acc_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(result_path + "ACC.png")
    plt.clf()

    Test = plt.plot(test_rmse_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(result_path + "RMSE.png")
    plt.clf()

    Test = plt.plot(test_mae_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(result_path + "MAE.png")
    plt.clf()

    Test = plt.plot(test_avg_loglike_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test AVG likelihood')
    plt.legend()
    plt.savefig(result_path + "AVG.png")
    plt.clf()


def make_records(result_path,test_overall_acc_list,test_aye_acc_list,test_nay_acc_list,test_notvoting_acc_list,
                     test_overall_RMSE_list,test_aye_RMSE_list,test_nay_RMSE_list,test_notvoting_RMSE_list,
                     test_overall_MAE_list,test_aye_MAE_list,test_nay_MAE_list,test_notvoting_MAE_list,test_avg_loglike_list,current_time,
                 args,model_name,data_name,train_ratio,hidden_neuron,random_seed,optimizer_method,lr):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    overview = '../results/' + 'overview.txt'
    basic_info = result_path + "basic_info.txt"
    test_record = result_path + "test_record.txt"

    with open(test_record, 'w') as g:

        ''' ACC '''
        g.write(str("Overall_ACC:"))
        g.write('\t')
        for itr in range(len(test_overall_acc_list)):
            g.write(str(test_overall_acc_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("AYE_ACC:"))
        g.write('\t')
        for itr in range(len(test_aye_acc_list)):
            g.write(str(test_aye_acc_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("NAY_ACC:"))
        g.write('\t')
        for itr in range(len(test_nay_acc_list)):
            g.write(str(test_nay_acc_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("Not_Voting_ACC:"))
        g.write('\t')
        for itr in range(len(test_notvoting_acc_list)):
            g.write(str(test_notvoting_acc_list[itr]))
            g.write('\t')
        g.write('\n')

        ''' RMSE '''
        g.write(str("Overall_RMSE:"))
        g.write('\t')
        for itr in range(len(test_overall_RMSE_list)):
            g.write(str(test_overall_RMSE_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("AYE_RMSE:"))
        g.write('\t')
        for itr in range(len(test_aye_RMSE_list)):
            g.write(str(test_aye_RMSE_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("NAY_RMSE:"))
        g.write('\t')
        for itr in range(len(test_nay_RMSE_list)):
            g.write(str(test_nay_RMSE_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("Not_Voting_RMSE:"))
        g.write('\t')
        for itr in range(len(test_notvoting_RMSE_list)):
            g.write(str(test_notvoting_RMSE_list[itr]))
            g.write('\t')
        g.write('\n')

        ''' MAE '''
        g.write(str("Overall_MAE:"))
        g.write('\t')
        for itr in range(len(test_overall_MAE_list)):
            g.write(str(test_overall_MAE_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("AYE_MAE:"))
        g.write('\t')
        for itr in range(len(test_aye_MAE_list)):
            g.write(str(test_aye_MAE_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("NAY_MAE:"))
        g.write('\t')
        for itr in range(len(test_nay_MAE_list)):
            g.write(str(test_nay_MAE_list[itr]))
            g.write('\t')
        g.write('\n')
        g.write(str("Not_Voting_MAE:"))
        g.write('\t')
        for itr in range(len(test_notvoting_MAE_list)):
            g.write(str(test_notvoting_MAE_list[itr]))
            g.write('\t')
        g.write('\n')


        g.write(str("AVG Likelihood:"))
        g.write('\t')
        for itr in range(len(test_avg_loglike_list)):
            g.write(str(test_avg_loglike_list[itr]))
            g.write('\t')
        g.write('\n')

    with open(basic_info, 'w') as h:
        h.write(str(args))

    with open(overview, 'a') as f:
        f.write(str(data_name))
        f.write('\t')
        f.write(str(model_name))
        f.write('\t')
        f.write(str(train_ratio))
        f.write('\t')
        f.write(str(current_time))
        f.write('\t')
        f.write(str(test_overall_RMSE_list[-1]))
        f.write('\t')
        f.write(str(test_overall_MAE_list[-1]))
        f.write('\t')
        f.write(str(test_overall_acc_list[-1]))
        f.write('\t')
        f.write(str(test_avg_loglike_list[-1]))
        f.write('\t')
        f.write(str(hidden_neuron))
        f.write('\t')
        f.write(str(args.corruption_level))
        f.write('\t')
        f.write(str(args.lambda_value))
        f.write('\t')
        f.write(str(args.lambda_t_value))
        f.write('\t')
        f.write(str(args.lambda_y))
        f.write('\t')
        f.write(str(args.lambda_u))
        f.write('\t')
        f.write(str(args.lambda_w))
        f.write('\t')
        f.write(str(args.lambda_n))
        f.write('\t')
        f.write(str(args.lambda_tau))
        f.write('\t')
        f.write(str(args.lambda_f))
        f.write('\t')
        f.write(str(args.lambda_alpha))
        f.write('\t')
        f.write(str(args.f_act))
        f.write('\t')
        f.write(str(args.g_act))
        f.write('\n')

    Test = plt.plot(test_overall_acc_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(result_path + "ACC.png")
    plt.clf()

    Test = plt.plot(test_overall_RMSE_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(result_path + "RMSE.png")
    plt.clf()

    Test = plt.plot(test_overall_MAE_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(result_path + "MAE.png")
    plt.clf()

    Test = plt.plot(test_avg_loglike_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test AVG likelihood')
    plt.legend()
    plt.savefig(result_path + "AVG.png")
    plt.clf()

def variable_save(result_path,model_name,train_var_list1,train_var_list2,train_var_list3,Estimated_R,test_v_ud,mask_test_v_ud):
    for var in train_var_list1:
        var_value = var.eval()
        var_name = (var.name.split(':'))[0]
        print (var_name)
        var_name = var_name.replace("/","_")
        #var_name = ((var.name).split('/'))[2]
        #var_name = (var_name.split(':'))[0]
        print (var.name)
        print (var_name)
        print ("================================")
        np.savetxt(result_path + var_name, var_value)

    for var in train_var_list2:
        var_value = var.eval()
        var_name = ((var.name).split('/'))[1]
        var_name = (var_name.split(':'))[0]
        np.savetxt(result_path + var_name , var_value)

    for var in train_var_list3:
        var_value = var.eval()
        var_name = (var.name.split(':'))[0]
        print (var_name)
        var_name = var_name.replace("/","_")
        #var_name = ((var.name).split('/'))[2]
        #var_name = (var_name.split(':'))[0]
        print (var.name)
        print (var_name)
        print ("================================")
        np.savetxt(result_path + var_name, var_value)

    np.savetxt(result_path + "raw_Estimated_R", Estimated_R)
    Estimated_R = np.where(Estimated_R<0.5,0,1)
    Error_list = np.nonzero( (Estimated_R - test_v_ud) * mask_test_v_ud )
    user_error_list = Error_list[0]
    item_error_list = Error_list[1]
    np.savetxt(result_path+"Estimated_R",Estimated_R)
    np.savetxt(result_path+"test_v_ud",test_v_ud)
    np.savetxt(result_path+"mask_test_v_ud",mask_test_v_ud)
    np.savetxt(result_path + "user_error_list", user_error_list)
    np.savetxt(result_path + "item_error_list", item_error_list)

def SDAE_calculate(model_name,X_c, layer_structure, W, b, batch_normalization, f_act,g_act, model_keep_prob,V_u=None):
    hidden_value = X_c
    for itr1 in range(len(layer_structure) - 1):
        ''' Encoder '''
        if itr1 <= int(len(layer_structure) / 2) - 1:
            if (itr1 == 0) and (model_name == "CDAE"):
                ''' V_u '''
                before_activation = tf.add(tf.add(tf.matmul(hidden_value, W[itr1]),V_u), b[itr1])
            else:
                before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = f_act(before_activation)
            ''' Decoder '''
        elif itr1 > int(len(layer_structure) / 2) - 1:
            before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = g_act(before_activation)
        if itr1 < len(layer_structure) - 2: # add dropout except final layer
            hidden_value = tf.nn.dropout(hidden_value, model_keep_prob)
        if itr1 == int(len(layer_structure) / 2) - 1:
            Encoded_X = hidden_value

    sdae_output = hidden_value

    return Encoded_X, sdae_output

def VAE_calculate(sess,X_c,layer_structure,W,b,do_batch_norm,nipen_activation,nipen_activation2,model_keep_prob,lambda_w,lambda_n,num_eps_samples):

    class Dense():
        """Fully-connected layer"""

        def __init__(self, scope="dense_layer", size=None, dropout=1.,
                     nonlinearity=tf.identity):
            # (str, int, (float | tf.Tensor), tf.op)
            assert size, "Must specify layer size (num nodes)"
            self.scope = scope
            self.size = size
            self.dropout = dropout  # keep_prob
            self.nonlinearity = nonlinearity

        def __call__(self, x):
            """Dense layer currying, to apply layer to any input tensor `x`"""
            # tf.Tensor -> tf.Tensor
            with tf.name_scope(self.scope):
                while True:
                    try:  # reuse weights if already initialized
                        return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                    except(AttributeError):
                        self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                        self.w = tf.nn.dropout(self.w, self.dropout)

        @staticmethod
        def wbVars(fan_in, fan_out):
            """Helper to initialize weights and biases, via He's adaptation
            of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
            """
            # (int, int) -> (tf.Variable, tf.Variable)
            stddev = tf.cast((2 / fan_in) ** 0.5, tf.float32)

            initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
            initial_b = tf.zeros([fan_out])

            with tf.variable_scope("VAE_Variable"):
                return_w = tf.Variable(initial_w, trainable=True, name="VAE_weights_w")
                retrun_b = tf.Variable(initial_b, trainable=True, name="VAE_weights_b")
            return return_w,retrun_b

    def composeAll(*args):
        """Util for multiple function composition

        i.e. composed = composeAll([f, g, h])
             composed(x) # == f(g(h(x)))
        """
        # adapted from https://docs.python.org/3.1/howto/functional.html
        return partial(functools.reduce, compose)(*args)

    def crossEntropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            return tf.nn.l2_loss(obs - actual) * lambda_n
            # obs_ = tf.clip_by_value(obs, , 1 - offset)
            # actual_ = tf.clip_by_value(actual,0,1)
            # return -1 * tf.reduce_sum(actual_ * tf.log(obs_) +
            #                       (1 - actual_) * tf.log(1 - obs_), 1)
            #return -1 * tf.reduce_sum(actual * tf.log(obs_) +
            #                      (1 - actual) * tf.log(1 - obs_), 1)

    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    def kullbackLeibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu ** 2 -
                                        tf.exp(2 * log_sigma), 1)

    def sampleGaussian(mu, log_sigma):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick

            a = (tf.shape(log_sigma))[0]
            b = (tf.shape(log_sigma))[1]

            #epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            epsilon = tf.reduce_mean(tf.random_normal((a,b,num_eps_samples), name="epsilon"))
            return mu + epsilon * tf.exp(log_sigma)  # N(mu, I * sigma**2)




    '''
    VAE part for encoder for representation (bottle-neck) and
    reconstructed output
    '''
    lambda_l2_reg = lambda_w
    # vae_dropout = tf.placeholder_with_default(0.9, shape=[], name='dropout')

    vae_hidden_activation = nipen_activation
    vae_squashing_activation = tf.identity
    # vae_architecture = [num_voca, 1000, 1000, num_topic]
    vae_architecture = layer_structure[0:int(round(len(layer_structure) / float(2)))]

    """(Re)build a symmetric VAE model with given:
        * architecture (list of nodes per encoder layer); e.g.
           [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
           & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]
    """
    # encoding q(z|x)

    encoding = [Dense('encoding', hidden_size, model_keep_prob, vae_hidden_activation)
                for hidden_size in reversed(vae_architecture[1:-1])]
    h_encoded = composeAll(encoding)(X_c)

    z_mean = Dense('z_mean', vae_architecture[-1], model_keep_prob)(h_encoded)
    z_log_sigma = Dense('z_log_sigma', vae_architecture[-1], model_keep_prob)(h_encoded)

    z = sampleGaussian(z_mean, z_log_sigma)

    # decoding p(x|z)
    decoding = [Dense('decoding', hidden_size, model_keep_prob, vae_hidden_activation)
                for hidden_size in vae_architecture[1:-1]]

    decoding.insert(0, Dense(  # prepend as outermost function
        "x_decoding", vae_architecture[0], model_keep_prob, vae_squashing_activation))

    x_hat = tf.identity(composeAll(decoding)(z), name='x_reconstructed')

    rec_loss = crossEntropy(x_hat, X_c)
    kl_loss = kullbackLeibler(z_mean, z_log_sigma)

    with tf.name_scope('l2_regularization'):
        regularizers = [tf.nn.l2_loss(var) for var in sess.graph.get_collection(
            "trainable_variables") if "weights" in var.name]
        l2_reg = lambda_l2_reg * tf.add_n(regularizers)

    ''' VAE cost'''
    with tf.name_scope('cost'):
        vae_cost = tf.reduce_mean(rec_loss + kl_loss, name='vae_cost')
        #vae_cost = tf.reduce_mean(rec_loss, name='vae_cost')


    return z , vae_cost,l2_reg



def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

def softmax(w, t = 1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist