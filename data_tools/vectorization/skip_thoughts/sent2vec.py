from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

# set paths to the model
VOCAB_FILE = '/skip_thoughts/skip_thoughts_bi/vocab.txt'
EMBEDDING_MATRIX_FILE = '/skip_thoughts/skip_thoughts_bi/embdeddings.npy'
CHECKPOINT_PATH = '/skoip_thoughts/skip_thoughts_bi/model.ckpt-500008.data'
DATA_DIR = '/skip_thoughts/skip_thoughts_bi/'

# set up the encoder. using a bidirectional model
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=True),
	vocabulary_file=VOCAB_FILE,
	embedding_matrix_file=EMBEDDING_MATRIX_FILE,
	checkpoint_path=CHECKPOINT_PATH)

# load the dataset
data = []
with open(os.path.join(DATA_DIR, 'rt-polarity.neg'), 'rb') as f:
  data.extend([line.decode('latin-1').strip() for line in f])
with open(os.path.join(DATA_DIR, 'rt-polarity.pos'), 'rb') as f:
  data.extend([line.decode('latin-1').strip() for line in f])
  
# generate skip-though vectors for each sentence in the dataset
encodings = encoder.encode(data)

print(encodings)