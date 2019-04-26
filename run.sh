#!/usr/bin/env bash

# Autorec
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=Autorec --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --corruption_level=0.0 \
--f_act=Sigmoid --g_act=Identity --hidden_neuron=100 --train_epoch=2000
done
done

# CDAE
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=CDAE --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --corruption_level=0.4 \
--f_act=Sigmoid --g_act=Sigmoid --hidden_neuron=50 --train_epoch=2000
done
done

# TrustSVD
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=TrustSVD --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --f_act=Sigmoid --g_act=Sigmoid \
--lambda_alpha=1.0 --lambda_f=10.0 --lambda_n=100.0 --lambda_t_value=1.0 --lambda_u=0.01 --lambda_v=100.0 --lambda_value=0.01 \
--lambda_w=1.0 --lambda_y=1.0 --use_network_positive=False --corruption_level=0.0 --train_epoch=2000
done
done

# CDL
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=CDL --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --f_act=Sigmoid --g_act=Sigmoid \
--lambda_alpha=1.0 --lambda_f=10.0 --lambda_n=100.0 --lambda_t_value=1.0 --lambda_u=0.01 --lambda_v=100.0 --lambda_value=0.01 \
--lambda_w=1.0 --lambda_y=1.0 --use_network_positive=False --train_epoch=2000
done
done

# NIPEN PGM(SDAE)
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=NIPEN --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --lambda_alpha=1.0 --lambda_f=10.0 \
--lambda_n=1000.0 --lambda_t_value=1.0 --lambda_tau=1.0 --lambda_u=0.1 --lambda_v=1.0 --lambda_value=0.01 \
--lambda_w=0.1 --lambda_y=1.0 --use_network_positive=False --train_epoch=2000
done
done

# NIPEN PGM(VAE approx.)
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=NIPEN --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --lambda_alpha=1.0 --lambda_f=10.0 \
--lambda_n=1000.0 --lambda_t_value=1.0 --lambda_tau=1.0 --lambda_u=0.1 --lambda_v=1.0 --lambda_value=0.01 \
--lambda_w=0.1 --lambda_y=1.0 --use_network_positive=False --network_split=True --G=3 --train_epoch=2000
done
done

# NIPEN PGM(VAE)
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=NIPEN_with_VAE --data=${itr1} --encoder_method=VAE --test_fold=${itr0} --lambda_alpha=1.0 --lambda_f=10.0 \
--lambda_n=1000.0 --lambda_t_value=1.0 --lambda_tau=1.0 --lambda_u=0.1 --lambda_v=1.0 --lambda_value=0.01 \
--lambda_w=0.1 --lambda_y=10.0 --use_network_positive=False --train_epoch=2000
done
done

# NIPEN Tensor
for itr0 in 0 1 2 3 4
do
for itr1 in politic_old politic_new
do
python3 main.py --model_name=NIPEN_tensor_single --data=${itr1} --encoder_method=VAE --test_fold=${itr0} --lambda_alpha=10.0 --lambda_f=10.0 \
--lambda_n=1000.0 --lambda_t_value=1.0 --lambda_tau=1000.0 --lambda_u=0.1 --lambda_v=1.0 --lambda_value=0.01 \
--lambda_w=10.0 --lambda_y=1000.0 --use_network_positive=False --train_epoch=2000
done
done