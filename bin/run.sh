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
# python train_team.py -m 'catboost_514' -c 'custom_514'
# python train_team.py -m 'catboost_515' -c 'custom_515'
# python train_team.py -m 'catboost_516' -c 'custom_516'


# ======================================================
# NN
# ======================================================
# python train_nn.py -m 'transformer_001' -c 'test'
# python train_nn.py -m 'transformer_002' -c 'test saint model'
# python train_nn.py -m 'transformer_003' -c 'dim_model=256'
# python train_nn.py -m 'transformer_004' -c 'num_en, num_de=3'
# python train_nn.py -m 'transformer_005' -c 'dim_model=512'
# python train_nn.py -m 'transformer_006' -c 'saint_v2, add difftime'
# python train_nn.py -m 'transformer_007' -c '006, lr=0.01'
# python train_nn.py -m 'transformer_008' -c '006, lr=0.0001'
# python train_nn.py -m 'transformer_009' -c 'saint_v3'
# python train_nn.py -m 'transformer_010' -c '003, seq=120'
# python train_nn.py -m 'transformer_011' -c '003, seq=150'
# python train_nn.py -m 'transformer_012' -c '010, epoch=50'
# python train_nn.py -m 'transformer_013' -c 'saint_v3'
# python train_nn.py -m 'transformer_014' -c '013, AdamW'
# python train_nn.py -m 'transformer_015' -c '013, batch_size=256'
# python train_nn.py -m 'transformer_016' -c 'saint_v4'
# python train_nn.py -m 'transformer_017' -c 'saint_v5, add prior_exp'
# python train_nn.py -m 'transformer_018' -c '017, SmoothBCEwLogits'
# python train_nn.py -m 'transformer_019' -c '017, num_en=3'
python train_nn.py -m 'transformer_020' -c 'saint_v5, add task_container_id'

# python train_nn_team.py -m 'mlp_001' -c 'test'
# python train_nn_team.py -m 'mlp_002' -c 'custom_512, epoch=30'
# python train_nn_team.py -m 'mlp_003' -c 'custom_512, layer_num=3'
# python train_nn_team.py -m 'mlp_004' -c 'custom_513'
# python train_nn_team.py -m 'mlp_005' -c 'custom_513'
# python train_nn_team.py -m 'mlp_006' -c 'custom_513, drop=0.1'
# python train_nn_team.py -m 'mlp_007' -c 'custom_513, drop=0.0'
# python train_nn_team.py -m 'mlp_008' -c 'custom_513, drop bn'
# python train_nn_team.py -m 'mlp_009' -c 'custom_513, PReLU'
# python train_nn_team.py -m 'mlp_010' -c 'custom_513'
# python train_nn_team.py -m 'mlp_011' -c 'custom_514'
# python train_nn_team.py -m 'mlp_012' -c 'custom_514, embedding'
# python train_nn_team.py -m 'mlp_013' -c 'custom_514'

# python train_nn_team.py -m 'tabnet_001' -c 'custom_514'


# ======================================================
# ENSEMBLE
# ======================================================
# python ensemble_team.py -m 'ensemble_001' -c 'test'
# python ensemble_team.py -m 'ensemble_002' -c 'rank=False'
# python ensemble_team.py -m 'ensemble_003' -c 'cb x1, mlp x1'
# python ensemble_team.py -m 'ensemble_004' -c 'cb x1, mlp x1'
# python ensemble_team.py -m 'ensemble_005' -c 'cb x1, mlp x1'