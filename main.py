from data_preprocessor import *
from models.NIPEN import NIPEN
from models.NIPEN_tensor_single import NIPEN_tensor_single
from models.CDAE import CDAE
from models.TrustSVD import TrustSVD
from models.CDL import CDL
from models.DAE import DAE
from models.AutoRec import AutoRec
import tensorflow as tf
import time
import argparse

current_time = time.time()

parser = argparse.ArgumentParser(description='NIPEN baseline Experiments')
parser.add_argument('--model_name', choices=['Autorec','CDL','TrustSVD','CDAE','NIPEN','NIPEN_with_VAE','NIPEN_tensor_single'], default='NIPEN_tensor_single')
parser.add_argument('--data_name', choices=['politic_old','politic_new'], default='politic_new')
parser.add_argument('--test_fold', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--train_epoch', type=int, default=2000)
parser.add_argument('--display_step', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp','GradientDescent','Momentum'],default='Adam')
parser.add_argument('--keep_prob', type=float, default=0.9)
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=0)
parser.add_argument('--grad_clip', choices=['True', 'False'], default='True')  # True
parser.add_argument('--batch_normalization', choices=['True','False'], default = 'False')

parser.add_argument('--hidden_neuron', type=int, default=10)
parser.add_argument('--corruption_level', type=float, default=0.3)
parser.add_argument('--lambda_value', type=float, default=0.01)
parser.add_argument('--lambda_t_value', type=float, default=1)

parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Relu')
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Relu')

parser.add_argument('--encoder_method', choices=['SDAE','VAE'],default='VAE')
parser.add_argument('--network_structure', choices=['alpha','user_alpha','doc_alpha','user_doc_alpha','without_network'],default='user_alpha',\
                    help="alpha : just alpha and (1-alpha). shared variable for all users / distinct_alpha : sigmoid individually(contents,network) / "
                         "user_alpha : allocate alpha to each user(alpha:1*U matrix) / without_network : zero networkparts")

parser.add_argument('--network_split', choices=['True','False'], default='False', help='Network Split true false')
parser.add_argument('--bias_structure', choices=['user','doc','user_doc','user_doc_seperate'], default='doc', help='use user and document voting bias or not')
parser.add_argument('--use_bias_reg', choices=['True','False'],default='False', help="Wheter to use tau_uu or tau_kk(reduced dim)")
parser.add_argument('--use_network_positive', choices=['True','False'],default='False', help="Wheter to use tau_uu or tau_kk(reduced dim)")
parser.add_argument('--init_constant_alpha',type=float , default = 0.3)

parser.add_argument('--G',type=int , default = 10)
parser.add_argument('--lambda_v',type=float , default = 1) # xi_dk prior std / cost3

parser.add_argument('--lambda_y',type=float , default = 10) # xi_dk prior std / cost3 (10)
parser.add_argument('--lambda_u',type=float , default = 0.1) # x_uk prior std / cost5
parser.add_argument('--lambda_w',type=float , default = 10) # SDAE weight std. / weight , bias regularization / cost1
parser.add_argument('--lambda_n',type=float , default = 1000) # SDAE output (cost2)
parser.add_argument('--lambda_tau',type=float , default = 1000) # tau_uu prior std / cost6
parser.add_argument('--lambda_f',type=float , default = 10) # voting result prior std / cost4
parser.add_argument('--lambda_alpha',type=float , default = 1000) # to regularize alpha matrix
parser.add_argument('--lambda_ntn',type=float , default = 1000) # to regularize alpha matrix
parser.add_argument('--lambda_not_voting',type=float , default = 1000) # to regularize alpha matrix
parser.add_argument('--consider_not_voting',choices=['True','False'], default='False', help='consider Wheter voting or not')
parser.add_argument('--wide_ntn',choices=['True','False'], default='False', help='consider Wheter voting or not')
args = parser.parse_args()

random_seed = args.random_seed
tf.reset_default_graph()
np.random.seed(random_seed)
# np.random.RandomState
tf.set_random_seed(random_seed)
network_split = args.network_split

model_name = args.model_name
data_name = args.data_name
data_base_dir = "data/"
path = data_base_dir + "%s" % data_name + "/"

if data_name == 'politic_new':
    num_users = 1537
    num_items = 7975
    num_total_ratings = 2999844
    num_voca = 13581
elif data_name == 'politic_old':
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703
    num_voca = 10000
else:
    raise NotImplementedError("ERROR")

a = args.a
b = args.b

test_fold = args.test_fold
hidden_neuron = args.hidden_neuron

keep_prob = args.keep_prob
batch_normalization = args.batch_normalization

batch_size = 128 #256
lr = args.lr
train_epoch = args.train_epoch
optimizer_method = args.optimizer_method
display_step = args.display_step
decay_epoch_step = 10000
decay_rate = 0.999
grad_clip = args.grad_clip

if args.f_act == "Sigmoid":
    f_act = tf.nn.sigmoid
elif args.f_act == "Relu":
    f_act = tf.nn.relu
elif args.f_act == "Tanh":
    f_act = tf.nn.tanh
elif args.f_act == "Identity":
    f_act = tf.identity
elif args.f_act == "Elu":
    f_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

if args.g_act == "Sigmoid":
    g_act = tf.nn.sigmoid
elif args.g_act == "Relu":
    g_act = tf.nn.relu
elif args.g_act == "Tanh":
    g_act = tf.nn.tanh
elif args.g_act == "Identity":
    g_act = tf.identity
elif args.g_act == "Elu":
    g_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

G = args.G

date = "0203"
result_path = 'results/' + data_name + '/' + model_name + '/' + str(test_fold) +  '/' + str(current_time)+"/"

R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, data_name,num_users, num_items,num_total_ratings, a, b, test_fold,random_seed,args.consider_not_voting)

X_dw = read_bill_term(path,data_name,num_items,num_voca)

print ("Type of Model : %s" %model_name)
print ("Type of Data : %s" %data_name)
print ("# of User : %d" %num_users)
print ("# of Item : %d" %num_items)
print ("Test Fold : %d" %test_fold)
print ("Random seed : %d" %random_seed)
print ("Hidden neuron : %d" %hidden_neuron)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:

    if model_name == "CDAE":
        lambda_value = args.lambda_value
        corruption_level = args.corruption_level

        #layer_structure = [num_items, hidden_neuron, num_items]
        layer_structure = [num_items, 512, 128, hidden_neuron, 128, 512, num_items]
        n_layer = len(layer_structure)
        pre_W = dict()
        pre_b = dict()
        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid")
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()

        model = CDAE(sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                    num_users,num_items,hidden_neuron,f_act,g_act,
                    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                    train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                    decay_epoch_step,lambda_value,
                    user_train_set, item_train_set, user_test_set, item_test_set,
                    result_path,date,data_name,model_name,test_fold,corruption_level)

    elif model_name == "Autorec":
        lambda_value = args.lambda_value

        layer_structure = [num_items, 512, 128, hidden_neuron, 128,512, num_items]
        #layer_structure = [num_items, 500,250,500, num_items]
        n_layer = len(layer_structure)
        pre_W = dict()
        pre_b = dict()
        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid")
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()

        model = AutoRec(sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                        num_users,num_items,hidden_neuron,f_act,g_act,
                        R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                        train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                        decay_epoch_step,lambda_value,
                        user_train_set, item_train_set, user_test_set, item_test_set,
                        result_path,date,data_name,model_name,test_fold)

    elif model_name == "TrustSVD":
        lambda_value = args.lambda_value
        lambda_t_value = args.lambda_t_value
        lambda_list = [lambda_value, lambda_t_value]

        trust_matrix = read_trust(path, data_name, num_users)
        model = TrustSVD(sess,args,
                         num_users,num_items,hidden_neuron,current_time,
                         R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,trust_matrix,
                         train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                         decay_epoch_step,lambda_value,
                         user_train_set, item_train_set, user_test_set, item_test_set,
                         result_path,date,data_name,
                         lambda_list,test_fold,model_name)

    elif model_name == "CDL":
        lambda_u = args.lambda_u
        lambda_v = args.lambda_v
        lambda_w = args.lambda_w
        lambda_n = args.lambda_n
        lambda_list = [lambda_u, lambda_w, lambda_v, lambda_n]
        corruption_level = args.corruption_level

        layer_structure = [num_voca, 512, 128, hidden_neuron, 128, 512, num_voca]
        n_layer = len(layer_structure)
        pre_W = dict()
        pre_b = dict()

        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid")
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()
        model = CDL(sess, num_users, num_items, num_voca, hidden_neuron,current_time,
                  batch_size, lambda_list, layer_structure, train_epoch,
                  pre_W, pre_b, f_act,g_act,
                  corruption_level, keep_prob,
                  num_train_ratings, num_test_ratings,
                  X_dw, R, train_R, test_R, C,
                  mask_R, train_mask_R, test_mask_R,
                  grad_clip, display_step, a, b,
                  optimizer_method, lr,
                  result_path,
                  decay_rate, decay_epoch_step, args, random_seed, model_name, test_fold,data_name)

    elif (model_name == "NIPEN") or (model_name == "NIPEN_without_network") or (model_name == "NIPEN_with_VAE") \
            or (model_name =="NIPEN_with_VAE_without_network") or (model_name == "NIPEN_only_network"):
        ############# Lambda ################

        lambda_y =  tf.cast(args.lambda_y,tf.float32)
        lambda_u =  tf.cast(args.lambda_u,tf.float32)
        lambda_w =  tf.cast(args.lambda_w,tf.float32)
        lambda_n =  tf.cast(args.lambda_n,tf.float32)
        lambda_tau =  tf.cast(args.lambda_tau,tf.float32)
        lambda_f =  tf.cast(args.lambda_f,tf.float32)
        lambda_alpha =  tf.cast(args.lambda_alpha,tf.float32)
        lambda_ntn = tf.cast(args.lambda_not_voting,tf.float32)

        lambda_penalty = tf.constant(1e20, dtype=tf.float32)  # # to make alpha positive
        lambda_list = [lambda_y, lambda_u, lambda_w, lambda_n, lambda_tau, lambda_f,
                       lambda_alpha,lambda_ntn,lambda_penalty]
        encoder_method = args.encoder_method
        if (model_name == "NIPEN_without_network") or (model_name == "NIPEN_with_VAE_without_network"):
            network_structure = "without_network"
        elif (model_name == "NIPEN_only_network"):
            network_structure = "only_network"
        else:
            network_structure = args.network_structure

        if (model_name == "NIPEN_with_VAE") or (model_name == "NIPEN_with_VAE_without_network"):
            encoder_method = "VAE"
            corruption_level = 0
        else:
            encoder_method = "SDAE"
        bias_structure = args.bias_structure
        use_bias_reg = args.use_bias_reg
        corruption_level = args.corruption_level
        init_constant_alpha = args.init_constant_alpha
        use_network_positive = args.use_network_positive

        layer_structure = [num_voca, 512, 128, hidden_neuron, 128, 512, num_voca]
        n_layer = len(layer_structure)
        pre_W = dict()
        pre_b = dict()
        trust_matrix = read_trust(path, data_name, num_users)

        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid",lambda_w)
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()


        model = NIPEN(sess,args,model_name,display_step,current_time,data_name,test_fold,random_seed,
                      num_items,num_voca,num_users,hidden_neuron,num_train_ratings,num_test_ratings,pre_W,pre_b, \
                      batch_size, train_epoch, lr, optimizer_method, decay_rate, \
                      f_act,g_act, encoder_method, batch_normalization, keep_prob,corruption_level,network_structure,\
                      layer_structure,bias_structure,use_bias_reg,
                      X_dw, train_R, test_R, train_mask_R, test_mask_R, trust_matrix, lambda_list,
                      init_constant_alpha,result_path,use_network_positive,
                      network_split,G)

    elif model_name == "NIPEN_tensor_single":
        ############# Lambda ################

        lambda_y =  tf.cast(args.lambda_y,tf.float32)
        lambda_u =  tf.cast(args.lambda_u,tf.float32)
        lambda_w =  tf.cast(args.lambda_w,tf.float32)
        lambda_n =  tf.cast(args.lambda_n,tf.float32)
        lambda_tau =  tf.cast(args.lambda_tau,tf.float32)
        lambda_f =  tf.cast(args.lambda_f,tf.float32)
        lambda_alpha =  tf.cast(args.lambda_alpha,tf.float32)
        lambda_ntn = tf.cast(args.lambda_ntn,tf.float32)

        lambda_penalty = tf.constant(1e20, dtype=tf.float32)  # # to make alpha positive
        lambda_list = [lambda_y, lambda_u, lambda_w, lambda_n, lambda_tau, lambda_f,
                       lambda_alpha,lambda_ntn,lambda_penalty]
        encoder_method = args.encoder_method
        if (model_name == "NIPEN_without_network") or (model_name == "NIPEN_with_VAE_without_network"):
            network_structure = "without_network"
        elif (model_name == "NIPEN_only_network"):
            network_structure = "only_network"
        else:
            network_structure = args.network_structure

        if (model_name == "NIPEN_with_VAE") or (model_name == "NIPEN_with_VAE_without_network"):
            encoder_method = "VAE"
            corruption_level = 0
        else:
            encoder_method = "VAE"
        bias_structure = args.bias_structure
        use_bias_reg = args.use_bias_reg
        corruption_level = args.corruption_level
        init_constant_alpha = args.init_constant_alpha
        use_network_positive = args.use_network_positive

        layer_structure = [num_voca, 512, 128, hidden_neuron, 128, 512, num_voca]
        n_layer = len(layer_structure)
        pre_W = dict()
        pre_b = dict()
        trust_matrix = read_trust(path, data_name, num_users)

        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid",lambda_w)
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()

        model = NIPEN_tensor_single(sess,args,model_name,display_step,current_time,data_name,test_fold,random_seed,
                      num_items,num_voca,num_users,hidden_neuron,num_train_ratings,num_test_ratings,pre_W,pre_b, \
                      batch_size, train_epoch, lr, optimizer_method, decay_rate, \
                      f_act,g_act, encoder_method, batch_normalization, keep_prob,corruption_level,network_structure,\
                      layer_structure,bias_structure,use_bias_reg,
                      X_dw, train_R, test_R, train_mask_R, test_mask_R, trust_matrix, lambda_list,
                      init_constant_alpha,result_path,use_network_positive,
                      network_split,G)

    else:
        raise NotImplementedError("ERROR")

    model.run()