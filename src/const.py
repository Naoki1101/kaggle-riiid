from pathlib import Path

INPUT_DATA_DIR = Path('../data/input')

ID_COLS = ['row_id']
TARGET_COLS = ['answered_correctly']

CATEGORICAL_COLS = [
    'user_id', 'content_id', 'content_type_id', 'task_container_id', 'prior_question_had_explanation'
]

NUMERICAL_COLS = [
    'timestamp', 'prior_question_elapsed_time'
]

DTYPE = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}

MAX_SEQ = 160
NUM_CONTENT = 13_523
