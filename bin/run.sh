#!/bin/bash

cd ../src

# ======================================================
# Table
# ======================================================
# python train.py -m 'lightgbm_001' -c 'test'
# python train.py -m 'lightgbm_002' -c 'custom_002'

# python train.py -m 'catboost_001' -c 'custom_002'
# python train.py -m 'catboost_002' -c 'custom_003'
# python train.py -m 'catboost_003' -c 'custom_003, depth=7'
# python train.py -m 'catboost_004' -c 'custom_003, depth=10'
# python train.py -m 'catboost_005' -c 'custom_004, depth=10'
# python train.py -m 'catboost_006' -c 'custom_005'
# python train.py -m 'catboost_007' -c 'custom_006'
# python train.py -m 'catboost_008' -c 'custom_007'
# python train.py -m 'catboost_009' -c 'custom_008'
# python train.py -m 'catboost_010' -c 'custom_009'
# python train.py -m 'catboost_011' -c 'custom_010'
# python train.py -m 'catboost_012' -c 'custom_011'
# python train.py -m 'catboost_013' -c 'custom_012'
# python train.py -m 'catboost_014' -c 'custom_013'
# python train.py -m 'catboost_015' -c 'custom_014'
# python train.py -m 'catboost_016' -c 'custom_015'
# python train.py -m 'catboost_017' -c 'custom_015, myaun params'
# python train.py -m 'catboost_018' -c 'custom_016'
# python train.py -m 'catboost_019' -c 'custom_017'
# python train.py -m 'catboost_020' -c 'custom_018'
# python train.py -m 'catboost_021' -c 'custom_018'
# python train.py -m 'catboost_022' -c 'custom_019'
# python train.py -m 'catboost_023' -c 'custom_020'
# python train.py -m 'catboost_024' -c 'custom_021'
# python train.py -m 'catboost_025' -c 'custom_022'
# python train.py -m 'catboost_026' -c 'custom_023'
# python train.py -m 'catboost_027' -c 'custom_024'
# python train.py -m 'catboost_028' -c 'custom_024, depth=10'
# python train.py -m 'catboost_029' -c 'custom_025'

# python train.py -m 'catboost_901' -c 'custom_901'

# python train_team.py -m 'catboost_501' -c 'custom_501'
# python train_team.py -m 'catboost_502' -c 'custom_502'
# python train_team.py -m 'catboost_503' -c 'custom_503'
# python train_team.py -m 'catboost_504' -c 'custom_504'
# python train_team.py -m 'catboost_505' -c 'custom_505'
# python train_team.py -m 'catboost_506' -c 'custom_506'
# python train_team.py -m 'catboost_507' -c 'custom_505'
# python train_team.py -m 'catboost_508' -c 'custom_507'
# python train_team.py -m 'catboost_509' -c 'custom_508'
# python train_team.py -m 'catboost_510' -c 'custom_509'
# python train_team.py -m 'catboost_511' -c 'custom_510'
# python train_team.py -m 'catboost_512' -c 'custom_511'
# python train_team.py -m 'catboost_513' -c 'custom_512'


# ======================================================
# NN
# ======================================================
# python train_nn.py -m 'transformer_001' -c 'test'

# python train_nn_team.py -m 'mlp_001' -c 'test'
# python train_nn_team.py -m 'mlp_002' -c 'custom_512, epoch=30'
# python train_nn_team.py -m 'mlp_003' -c 'custom_512, layer_num=3'
python train_nn_team.py -m 'mlp_004' -c 'custom_513'


# ======================================================
# ENSEMBLE
# ======================================================
# python ensemble_team.py -m 'ensemble_001' -c 'test'
# python ensemble_team.py -m 'ensemble_002' -c 'rank=False'