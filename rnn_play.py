# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import my_txtutils

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

# use topn=10 for all but the last one which works with topn=2 for Shakespeare and topn=3 for Python
author = "checkpoints/rnn_train_1528598304-9000000.data-00000-of-00001"

ncnt = 0

# tf.device('/device:GPU:0')

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1528601162.meta')
    # new_saver.restore(sess, author)

    # Suggested in https://github.com/martin-gorner/tensorflow-rnn-shakespeare/issues/4#issuecomment-299725416
    new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
    x = ord("L")
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    data = ''
    for i in range(4096):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        data += chr(c)

    print(data.encode("utf-8").hex())
