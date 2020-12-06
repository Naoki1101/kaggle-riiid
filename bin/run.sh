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
python train.py -m 'catboost_009' -c 'custom_008'


# ======================================================
# NN
# ======================================================
# python train_nn.py -m 'transformer_001' -c 'test'   # ERROR