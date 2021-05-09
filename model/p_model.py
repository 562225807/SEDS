import tensorflow as tf


class PWrapper:
    def __init__(self, word_embeddings, spectrogram, person_id, word_config, spectrogram_config,
                 lstm_int_num, is_train, mul_att_num=64, mul_num=8, name="person"):
        """
        :param word_embeddings [batch_size, max_time, embedding_length]
        :param spectrogram [batch_size, max_time, embedding_length]
        :param person_embeddings [batch_size, embedding_length]
        :param word_config: [layers_number, layer_setting]
        :param spectrogram_config: [layers_number, layer_setting]
        :param lstm_int_num: int (number of hidden layer units)
        """
        self.word_config = word_config
        self.spectrogram_config = spectrogram_config
        self.lstm_int_num = lstm_int_num
        self.word_embeddings = word_embeddings
        self.spectrogram_feature = spectrogram
        self.person_id = person_id
        self.mul_att_num = mul_att_num
        self.mul_num = mul_num
        self.latent_size = 128
        self.is_train = is_train
        self.name = name

    def sample_gaussian(self, mu, logvar):
        epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
        std = tf.exp(0.5 * logvar)
        z = mu + tf.multiply(std, epsilon)
        return z

    def _weight_variable(self, shape):
        #return tf.get_variable(name, shape=shape, initializer=tf.contrib.keras.initializers.he_normal())
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def identity_block(self, x_input, kernel_size, filters_number, layer_number):
        with tf.variable_scope("res_" + str(layer_number)):

            w1 = self._weight_variable([1, 1, filters_number, filters_number])
            conv = tf.nn.conv2d(x_input, w1, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.layers.batch_normalization(conv, training=self.is_train)
            conv = tf.nn.relu(conv)

            w2 = self._weight_variable([kernel_size, kernel_size, filters_number, filters_number])
            conv = tf.nn.conv2d(conv, w2, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.layers.batch_normalization(conv, training=self.is_train)
            conv = tf.nn.relu(conv)

            w3 = self._weight_variable([1, 1, filters_number, filters_number])
            conv = tf.nn.conv2d(conv, w3, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.layers.batch_normalization(conv, training=self.is_train)
            conv = tf.nn.relu(conv)

            output = x_input + conv
            output = tf.nn.relu(output)
            return output

    def _attention(self, word, spectrogram):
        word = tf.transpose(word, perm=[0, 3, 1, 2])
        spectrogram = tf.transpose(spectrogram, perm=[0, 3, 1, 2])

        wq = tf.transpose(tf.layers.dense(word, 1, use_bias=False), perm=[0, 1, 3, 2])
        wk = tf.layers.dense(spectrogram, 1, use_bias=False)
        b = self._bias_variable([1])
        att = tf.nn.tanh(wq + wk + b)
        att = tf.nn.softmax(logits=att, axis=1)
        att = tf.reshape(tf.reduce_sum(att, axis=-1), [-1, att.shape[1], att.shape[-2], 1])
        multimodal = spectrogram * att

        multimodal = tf.transpose(multimodal, perm=[0, 2, 3, 1])
        multimodal = tf.reshape(multimodal, [-1, int(multimodal.shape[1]), multimodal.shape[2] * multimodal.shape[3]])
        return multimodal

    def _multi_attention(self, person_embeddings, bi_out):
        result = []

        wq = [tf.layers.Dense(self.mul_att_num, use_bias=False) for _ in range(self.mul_num)]
        wk = [tf.layers.Dense(self.mul_att_num, use_bias=False) for _ in range(self.mul_num)]
        wv = [tf.layers.Dense(self.mul_att_num, use_bias=False) for _ in range(self.mul_num)]
        for _ in range(self.mul_num):
            Q = tf.reshape(wq[_](person_embeddings), [-1, self.mul_att_num, 1])
            K = wk[_](bi_out)
            V = wv[_](bi_out)
            att = tf.nn.softmax(tf.matmul(K, Q)/8, axis=1)
            result.append(tf.reduce_sum(V * att, axis=1))
        return tf.concat(result, 1)

    def memory_net(self, bi_out, query):
        with tf.variable_scope("emo_trans"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
            out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell, inputs=bi_out,
                                                         swap_memory=True, dtype=tf.float32)
        bi_out = tf.concat(out, 2)

        tmp = tf.reshape(query, [-1, int(query.shape[-1]), 1])
        att = tf.nn.softmax(tf.matmul(bi_out, tmp), axis=1)
        att = tf.reduce_sum(bi_out * att, axis=1)
        query = tf.nn.tanh(query + att)

        with tf.variable_scope("emo_trans2"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
            out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell, inputs=bi_out,
                                                         swap_memory=True, dtype=tf.float32)
        bi_out = tf.concat(out, 2)

        tmp = tf.reshape(query, [-1, int(query.shape[-1]), 1])
        att = tf.nn.softmax(tf.matmul(bi_out, tmp), axis=1)
        att = tf.reduce_sum(bi_out * att, axis=1)
        query = tf.nn.tanh(query + att)

        with tf.variable_scope("emo_trans3"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
            out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell, inputs=bi_out,
                                                         swap_memory=True, dtype=tf.float32)

        bi_out = tf.concat(out, 2)

        tmp = tf.reshape(query, [-1, int(query.shape[-1]), 1])
        att = tf.nn.softmax(tf.matmul(bi_out, tmp), axis=1)
        att = tf.reduce_sum(bi_out * att, axis=1)
        query = tf.nn.tanh(query + att)

        return query

    def multi_LSTM(self, bi_out):
        p_out = []
        for i in range(6):
            with tf.variable_scope(f"emo_trans{str(i)}"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell, inputs=bi_out,
                                                             swap_memory=True, dtype=tf.float32)
                out = tf.concat(out, 2)
            p_out.append(out[:, -1])
        p_out = tf.transpose(tf.stack(p_out), perm=[1, 0, 2])
        p_weight = tf.reshape(tf.one_hot(self.person_id, 6), [-1, 6, 1])
        p_out = tf.reduce_sum(p_out * p_weight, axis=1)
        print(p_out.shape)

        return p_out

    def CVAE(self, post_sum_state, response_sum_state):
        # recognition network
        with tf.variable_scope("recog_net"):
            recog_input = tf.concat([post_sum_state, response_sum_state], 1)
            recog_mulogvar = tf.contrib.layers.fully_connected(recog_input, self.latent_size * 2, activation_fn=None,
                                                               scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        # prior network
        with tf.variable_scope("prior_net"):
            prior_fc1 = tf.contrib.layers.fully_connected(tf.concat([post_sum_state, self.person_embeddings, self.s_emo_emb], 1),
                                                          self.latent_size * 2, activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = tf.contrib.layers.fully_connected(prior_fc1, self.latent_size * 2, activation_fn=None,
                                                               scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

        latent_sample = tf.cond(self.is_train,
                                lambda: self.sample_gaussian(recog_mu, recog_logvar),
                                lambda: self.sample_gaussian(prior_mu, prior_logvar))

        # Discriminator
        with tf.variable_scope("discriminator"):
            dis_input = latent_sample
            pattern_fc1 = tf.contrib.layers.fully_connected(dis_input, self.latent_size, activation_fn=tf.nn.relu,
                                                            scope="pattern_fc1")
            pattern_logits = tf.contrib.layers.fully_connected(pattern_fc1, self.emo_num, activation_fn=None,
                                                                    scope="pattern_logits")

        return recog_mu, recog_logvar, prior_mu, prior_logvar, pattern_logits

    def build_p(self):
        """
        :return: multi_modal_feature: [batch_size, feature_length, max_time]
        """
        with tf.variable_scope(self.name):
            word_inputs = tf.expand_dims(self.word_embeddings, -1)
            for _, layer in enumerate(self.word_config):
                with tf.variable_scope(f"word_{str(_)}"):
                    if layer[0] == 'conv':
                        weights = self._weight_variable([layer[1], layer[1], int(word_inputs.shape[-1]), layer[3]])
                        conv = tf.nn.conv2d(word_inputs, weights, strides=[1, layer[2], layer[2], 1], padding='SAME')
                        conv = tf.layers.batch_normalization(conv, training=self.is_train)
                        word_inputs = tf.nn.relu(conv)
                    elif layer[0] == 'pooling':
                        word_inputs = tf.nn.max_pool(word_inputs, ksize=[1, layer[1], layer[1], 1],
                                                     strides=[1, layer[2], layer[2], 1], padding='SAME')
                    else:
                        word_inputs = self.identity_block(word_inputs, layer[1], layer[2], _)

            spectrogram_inputs = tf.expand_dims(self.spectrogram_feature, -1)
            for _, layer in enumerate(self.spectrogram_config):
                with tf.variable_scope(f"spectrogram_{str(_)}"):
                    if layer[0] == 'conv':
                        weights = self._weight_variable([layer[1], layer[1], int(spectrogram_inputs.shape[-1]), layer[3]])
                        conv = tf.nn.conv2d(spectrogram_inputs, weights, strides=[1, layer[2], layer[2], 1], padding='SAME')
                        conv = tf.layers.batch_normalization(conv, training=self.is_train)
                        spectrogram_inputs = tf.nn.relu(conv)
                    elif layer[0] == 'pooling':
                        spectrogram_inputs = tf.nn.max_pool(spectrogram_inputs, ksize=[1, layer[1], layer[1], 1],
                                                     strides=[1, layer[2], layer[2], 1], padding='SAME')
                    else:
                        spectrogram_inputs = self.identity_block(spectrogram_inputs, layer[1], layer[2], _)

            multimodal = self._attention(word_inputs, spectrogram_inputs)

            # extract multi-modal feature
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_int_num, forget_bias=1.0)
            out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell, inputs=multimodal,
                                                         swap_memory=True, dtype=tf.float32)
            multimodal = tf.concat(out, 2)
            print("multimodal", multimodal.shape)

            # target encoder
            # with tf.variable_scope("target_encoder"):
            #     lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_int_num, forget_bias=1.0)
            #     out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell, inputs=self.t_embeddings,
            #                                                  swap_memory=True, dtype=tf.float32)
            # response_out = tf.concat([state[0][1], state[1][1]], 1)
            #
            # recog_mu, recog_logvar, prior_mu, prior_logvar, pattern_logits = self.CVAE(post_out, response_out)
            # select_cat = tf.argmax(tf.nn.softmax(pattern_logits, axis=-1), axis=-1)

            # dense layer
            # for dense_unit in self.dense_config:
            #     res_emo_fea = tf.layers.Dense(dense_unit, activation=tf.nn.relu)(res_emo_fea)
            #
            # dense = tf.layers.Dense(self.emo_num)(tf.layers.dropout(res_emo_fea, rate=self.drop_rate, training=self.is_train))
            #
            # logits = self.softargmax_beta * dense
            # emo_cat_prob = tf.nn.softmax(logits, axis=-1)
            # select_cat = tf.argmax(emo_cat_prob, axis=-1)

            # s_emo_logits = tf.layers.Dense(128, activation=tf.nn.relu)(post_out)
            # s_emo_logits = tf.layers.Dense(self.emo_num)\
            #     (tf.layers.dropout(s_emo_logits, rate=self.drop_rate, training=self.is_train))
            # t_emo_logits = tf.layers.Dense(128, activation=tf.nn.relu)(response_out)
            # t_emo_logits = tf.layers.Dense(self.emo_num) \
            #     (tf.layers.dropout(t_emo_logits, rate=self.drop_rate, training=self.is_train))

            # return multimodal, recog_mu, recog_logvar, prior_mu, prior_logvar, pattern_logits, select_cat, s_emo_logits\
            #     , t_emo_logits
            return multimodal
