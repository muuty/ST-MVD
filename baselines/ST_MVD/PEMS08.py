import os, sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../..'))
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.ST_MVD.arch import STMVD

DATA_NAME = 'PEMS08'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_PARAM = EasyDict({
    "num_nodes": 170,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    # Relational: K orthogonal fingerprint views
    "node_dim": 24, "num_views": 2, "fingerprint_seed": 42,
    # Temporal: ToD + DoW
    "if_T_i_D": True, "if_D_i_W": True,
    "temp_dim_tid": 32, "temp_dim_diw": 32,
    "time_of_day_size": 288, "day_of_week_size": 7,
    # Temporal: V_t parallel branches, each with a 3-layer MLP
    "num_temporal_branches": 4, "temporal_branch_dim": 16, "num_layer": 3,
})

NUM_EPOCHS = 100
CFG = EasyDict()
CFG.DESCRIPTION = 'ST-MVD on PEMS08'
CFG.GPU_NUM = 1
CFG.ENV = EasyDict({'SEED': 42, 'DETERMINISTIC': True})
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({'dataset_name': DATA_NAME, 'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
                              'input_len': INPUT_LEN, 'output_len': OUTPUT_LEN})
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({'dataset_name': DATA_NAME, 'train_ratio': TRAIN_VAL_TEST_RATIO[0],
                             'norm_each_channel': NORM_EACH_CHANNEL, 'rescale': RESCALE})
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'STMVD'
CFG.MODEL.ARCH = STMVD
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'MAPE': masked_mape, 'RMSE': masked_rmse})
CFG.METRICS.TARGET = 'MAE'; CFG.METRICS.NULL_VAL = NULL_VAL
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join('checkpoints', 'STMVD_PEMS08',
    '_'.join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict({'TYPE': 'Adam', 'PARAM': {'lr': 0.002, 'weight_decay': 0.0001}})
CFG.TRAIN.LR_SCHEDULER = EasyDict({'TYPE': 'MultiStepLR',
    'PARAM': {'milestones': [1, 20, 40, 60, 80], 'gamma': 0.5}})
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.DATA = EasyDict({'BATCH_SIZE': 32, 'SHUFFLE': True})
CFG.VAL = EasyDict({'INTERVAL': 1, 'DATA': EasyDict({'BATCH_SIZE': 64})})
CFG.TEST = EasyDict({'INTERVAL': 1, 'DATA': EasyDict({'BATCH_SIZE': 64})})
CFG.EVAL = EasyDict({'HORIZONS': [3, 6, 12], 'USE_GPU': True})
