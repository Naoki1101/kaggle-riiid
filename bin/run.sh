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
python train.py -m 'catboost_019' -c 'custom_017'


# ======================================================
# NN
# ======================================================
# python train_nn.py -m 'transformer_001' -c 'test'   # ERROR