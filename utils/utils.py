from model.encoder import build_encoder
from model.decoder import build_decoder, build_ECM_decoder, build_PEC_decoder
from model.CAN import CANWrapper
from model.personal_model import PMWrapper
from model.p_model import PWrapper
from model.CVAE import CVAEWrapper
from model.CAN_model import CANNewWrapper
from model.baseline import BaseWrapper
from model.new_model import NewWrapper

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import pickle


def init_embeddings(vocab_size, embed_size, dtype=tf.float32, initializer=None, initial_values=None,
                    name='embeddings'):
    """
    embeddings:
        initialize trainable embeddings or load pretrained from files
    """
    with tf.variable_scope(name):
        if initial_values is not None:
            embeddings = tf.Variable(initial_value=initial_values,
                                     name="embeddings", dtype=dtype)
        else:
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            embeddings = tf.Variable(
                initializer(shape=(vocab_size, embed_size)),
                name="embeddings", dtype=dtype)

        # id_0 represents SOS token, id_1 represents EOS token
        se_embed = tf.get_variable("SOS/EOS", [2, embed_size], dtype)
        # id_2 represents constant all zeros
        zero_embed = tf.zeros(shape=[1, embed_size])
        embeddings = tf.concat([se_embed, zero_embed, embeddings], axis=0)
    return embeddings


def compute_perplexity(sess, CE, mask, feed_dict):
    """
    Compute perplexity for a batch of data
    """
    CE_words = sess.run(CE, feed_dict=feed_dict)
    N_words = np.sum(mask)
    return np.exp(CE_words / N_words)


def compute_test_perplexity(sess, CE, mask, feed_dict):
    """
    Compute perplexity for a batch of data
    """
    CE_words = sess.run(CE, feed_dict=feed_dict)
    N_words = np.sum(mask)
    return CE_words, N_words


def get_PEC_config(config):
    enc_num_layers = config["encoder"]["num_layers"]
    enc_num_units = config["encoder"]["num_units"]
    enc_cell_type = config["encoder"]["cell_type"]
    enc_bidir = config["encoder"]["bidirectional"]

    dec_num_layers = config["decoder"]["num_layers"]
    dec_num_units = config["decoder"]["num_units"]
    dec_cell_type = config["decoder"]["cell_type"]
    state_pass = config["decoder"]["state_pass"]

    num_emo = config["decoder"]["num_emotions"]
    emo_cat_units = config["decoder"]["emo_cat_units"]
    emo_int_units = config["decoder"]["emo_int_units"]

    infer_batch_size = config["inference"]["infer_batch_size"]
    beam_size = config["inference"]["beam_size"]
    max_iter = config["inference"]["max_length"]
    attn_num_units = config["decoder"]["attn_num_units"]
    l2_regularize = config["training"]["l2_regularize"]

    conv_word_config = config["convolution"]["word"]
    word_config = []
    for layer in conv_word_config:
        if layer["type"] == "conv":
            word_config.append((layer["type"], layer["size"], layer["step"], layer["filter_num"]))
        elif layer["type"] == 'pooling':
            word_config.append((layer["type"], layer["size"], layer["step"]))
        else:
            word_config.append((layer["type"], layer["size"], layer["filter_num"]))

    conv_spec_config = config["convolution"]["spectrogram"]
    spectrogram_config = []
    for layer in conv_spec_config:
        if layer["type"] == "conv":
            spectrogram_config.append((layer["type"], layer["size"], layer["step"], layer["filter_num"]))
        elif layer["type"] == "pooling":
            spectrogram_config.append((layer["type"], layer["size"], layer["step"]))
        else:
            spectrogram_config.append((layer["type"], layer["size"], layer["filter_num"]))

    lstm_int_num = config["lstm"]["int_num"]

    batch_size = config["training"]["batch_size"]
    loss_weight = config["training"]["loss_weight"]
    return (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
            dec_num_layers, dec_num_units, dec_cell_type, state_pass,
            num_emo, emo_cat_units, emo_int_units, infer_batch_size,
            beam_size, max_iter, attn_num_units, l2_regularize, word_config,
            spectrogram_config, lstm_int_num, batch_size, loss_weight)


def compute_loss(source_ids, target_ids, sequence_mask, choice_qs,
                     embeddings, enc_num_layers, enc_num_units, enc_cell_type,
                     enc_bidir, dec_num_layers, dec_num_units, dec_cell_type,
                     state_pass, num_emo, emo_cat, emo_cat_units,
                     emo_int_units, infer_batch_size,
                     spectrogram, word_config, num_per, person_ids, per_cat_units,
                     spectrogram_config, loss_weight, lstm_int_num, is_train, is_pretrain,
                     lexicons_ids,
                     beam_size=None, max_iter=20, attn_num_units=128, l2_regularize=None,
                     name="PEC", is_0le=False):
    """
    Creates an ECM model and returns CE loss plus regularization terms.
        choice_qs: [batch_size, max_time], true choice btw generic/emo words
        emo_cat: [batch_size], emotion categories of each target sequence

    Returns
        CE: cross entropy, used to compute perplexity
        total_loss: objective of the entire model
        train_outs: (cell, log_probs, alphas, final_int_mem_states)
            alphas - predicted choices
        infer_outputs: namedtuple(logits, ids), [batch_size, max_time, d]
    """
    with tf.name_scope(name):
        # build encoder
        encoder_outputs, encoder_states = build_encoder(
            embeddings, source_ids, enc_num_layers, enc_num_units,
            enc_cell_type, bidir=enc_bidir, name="%s_encoder" % name)

        # gain lexicons embeddings
        if is_pretrain or is_0le:
            lexicons_embeddings = tf.zeros([embeddings.shape[0].value, embeddings.shape[1].value])
            lexicons_embeddings = tf.nn.embedding_lookup(lexicons_embeddings, lexicons_ids)
        else:
            lexicons_embeddings = tf.nn.embedding_lookup(embeddings, lexicons_ids)
        lex_emb = tf.nn.embedding_lookup(lexicons_embeddings, person_ids)

        # extract multi-modal feature
        word_embeddings = tf.nn.embedding_lookup(embeddings, source_ids)
        p_model = PWrapper(word_embeddings, spectrogram, person_ids, word_config, spectrogram_config,
                           lstm_int_num, is_train)

        multimodal_outputs = p_model.build_p()

        # build decoder: logits, [batch_size, max_time, vocab_size]
        cell, train_outputs, infer_outputs = build_PEC_decoder(
            encoder_outputs, encoder_states, multimodal_outputs, lex_emb, embeddings,
            num_per, person_ids, per_cat_units,
            dec_num_layers, dec_num_units, dec_cell_type,
            num_emo, emo_cat, emo_cat_units, emo_int_units,
            state_pass, infer_batch_size, attn_num_units,
            target_ids, beam_size, max_iter, is_pretrain,
            name="%s_decoder" % name)

        g_logits, e_logits, alphas, int_M_emo = train_outputs
        g_probs = tf.nn.softmax(g_logits) * (1 - alphas)
        e_probs = tf.nn.softmax(e_logits) * alphas
        train_log_probs = tf.log(g_probs + e_probs)

        with tf.name_scope('loss'):
            final_ids = tf.pad(target_ids, [[0, 0], [0, 1]], constant_values=1)
            alphas = tf.squeeze(alphas, axis=-1)
            choice_qs = tf.pad(choice_qs, [[0, 0], [0, 1]], constant_values=0)

            CE = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_log_probs, labels=final_ids)

            # compute losses

            g_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=g_logits, labels=final_ids) - tf.log(1 - alphas)

            e_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=e_logits, labels=final_ids) - tf.log(alphas)

            losses = g_losses * (1 - choice_qs) + e_losses * choice_qs

            # prepare for perplexity computations and person classifier
            target_embeddings = tf.nn.embedding_lookup(embeddings, final_ids)
            word_pros = -1 * tf.log(tf.reduce_sum(tf.one_hot(final_ids, int(train_log_probs.shape[-1])) *
                                      tf.nn.softmax(train_log_probs, axis=-1), axis=-1)[:, :-1])
            # word_pros = CE[:, :-1]
            Ewe = tf.reshape(word_pros, [-1, int(word_pros.shape[-1]), 1]) \
                  * tf.reshape(tf.cast(sequence_mask, tf.float32), [-1, int(sequence_mask.shape[-1]), 1]) \
                  * target_embeddings[:, :-1]
            classsifier = tf.layers.Dense(num_per, activation=None)
            Ewe = classsifier(tf.reduce_mean(Ewe, axis=1))

            # tmp_embeddings = tf.concat([embeddings, embeddings], 0)
            # infer_log_pro = infer_outputs.logits
            # infer_onehot = tf.one_hot(infer_outputs.ids, infer_outputs.logits.shape[-1].value)
            # infer_log_pro = -1 * tf.reduce_sum(infer_log_pro * infer_onehot, axis=-1)[:, :, 0]
            # infer_log_pro = tf.expand_dims(infer_log_pro, axis=-1)
            # infer_emb = tf.nn.embedding_lookup(tmp_embeddings, infer_outputs.ids[:, :, 0])
            # infer_emb = infer_log_pro * infer_emb
            # person_probs = tf.nn.softmax(classsifier(tf.reduce_mean(infer_emb, axis=1)), axis=-1)
            # person_onehot = tf.one_hot(person_ids, num_per)
            # score = tf.reduce_sum(person_probs * person_onehot, axis=-1)

            infer_ids = infer_outputs.ids[:, :, 0] % int(embeddings.shape[0])
            infer_emb = tf.nn.embedding_lookup(embeddings, infer_ids)
            infer_log_pro = tf.exp(infer_outputs.logits[:, :, 0])
            infer_gen_pro, infer_emo_pro = tf.split(infer_log_pro, [42003, 42003], -1)
            infer_pro = tf.log(infer_gen_pro + infer_emo_pro)
            infer_word_pros = -1 * tf.log(tf.reduce_sum(tf.one_hot(infer_ids, int(infer_pro.shape[-1])) *
                                      tf.nn.softmax(infer_pro, axis=-1), axis=-1))
            infer_sequence_mask = tf.not_equal(infer_ids, 1)

            infer_Ewe = tf.expand_dims(infer_word_pros, -1)\
                  * tf.expand_dims(tf.cast(infer_sequence_mask, tf.float32), -1) \
                  * infer_emb
            infer_Ewe = classsifier(tf.reduce_sum(infer_Ewe, axis=1)/40)
            score = tf.exp(-1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=infer_Ewe, labels=person_ids))

            # alpha and internal memory regularizations
            person_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=Ewe, labels=person_ids))

            alpha_reg = tf.reduce_mean(choice_qs * -tf.log(alphas))

            int_mem_reg = tf.reduce_mean(tf.norm(int_M_emo, axis=1))

            losses = tf.boolean_mask(losses[:, :-1], sequence_mask)
            if is_pretrain:
                reduced_loss = tf.reduce_mean(losses) + alpha_reg + int_mem_reg
            else:
                reduced_loss = tf.reduce_mean(losses) + alpha_reg + int_mem_reg + loss_weight * person_loss

            train_outs = (cell, train_log_probs, alphas, int_M_emo)

            CE = tf.boolean_mask(CE[:, :-1], sequence_mask)
            CE = tf.reduce_sum(CE)

            if l2_regularize is None:
                return CE, reduced_loss, loss_weight * person_loss, train_outs, infer_outputs, score
            else:
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not('bias' in v.name)])

                total_loss = reduced_loss + l2_regularize * l2_loss
                return CE, total_loss, loss_weight * person_loss, train_outs, infer_outputs, score


def get_training_config(config, field):
    train_config = config[field]
    logdir = train_config["logdir"]
    restore_from = train_config["restore_from"]

    learning_rate = train_config["learning_rate"]
    gpu_fraction = train_config["gpu_fraction"]
    max_checkpoints = train_config["max_checkpoints"]
    train_steps = train_config["train_steps"]
    batch_size = train_config["batch_size"]
    print_every = train_config["print_every"]
    checkpoint_every = train_config["checkpoint_every"]

    s_filename = train_config["train_source_file"]
    t_filename = train_config["train_target_file"]
    q_filename = train_config["train_choice_file"]
    sp_filename = train_config["spectrogram_file"]

    s_max_leng = train_config["source_max_length"]
    t_max_leng = train_config["target_max_length"]
    sp_max_leng = train_config["spectrograml_max_length"]

    dev_s_filename = train_config["dev_source_file"]
    dev_t_filename = train_config["dev_target_file"]
    dev_q_filename = train_config["dev_choice_file"]

    test_s_filename = train_config["test_source_file"]
    test_t_filename = train_config["test_target_file"]
    test_q_filename = train_config["test_choice_file"]
    test_output = train_config["test_output"]

    loss_fig = train_config["loss_fig"]
    perp_fig = train_config["perplexity_fig"]

    return (logdir, restore_from, learning_rate, gpu_fraction, max_checkpoints,
            train_steps, batch_size, print_every, checkpoint_every,
            s_filename, t_filename, q_filename,
            s_max_leng, t_max_leng, dev_s_filename, dev_t_filename,
            dev_q_filename, loss_fig, perp_fig, sp_filename, sp_max_leng,
            test_s_filename, test_t_filename, test_q_filename, test_output)


def load(saver, sess, logdir):
    """
    Load the latest checkpoint
    Ref: https://github.com/ibab/tensorflow-wavenet
    """
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def save(saver, sess, logdir, step):
    """
    Save the checkpoint
    """
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def loadfile(filename, is_dialog, is_source, max_length):
    with open(filename, encoding="utf-8") as f:
        sentence_ids = []
        persons = []
        dialogs = []
        emotions = []
        if is_dialog:
            for line in f.readlines():
                sentence_id, person, dialog, emotion = line.strip().split('\t')
                sentence_ids.append(sentence_id)
                persons.append(int(person))
                emotions.append(int(emotion))
                dialog = np.array(list(map(int, dialog.split())), dtype=np.int32)
                leng = len(dialog)
                if leng > max_length: continue
                if leng < max_length:
                    if is_source:
                        pads = -np.ones(max_length - leng, dtype=np.int32)
                        dialog = np.concatenate((pads, dialog))
                    else:
                        pads = -2 * np.ones(max_length - leng, dtype=np.int32)
                        dialog = np.concatenate((dialog, pads))
                dialogs.append(dialog)
            persons = np.array(persons, dtype=np.int32)
            emotions = np.array(emotions, dtype=np.int32)
            dialogs = np.array(dialogs, dtype=np.int32)
            if is_source:
                return sentence_ids, persons, dialogs
            else:
                return sentence_ids, persons, dialogs, emotions
        else:
            choices = []
            for line in f.readlines():
                choice = list(map(int, line.strip().split()))
                choice = np.array(choice)
                leng = len(choice)
                if leng > max_length: continue
                if leng < max_length:
                    pads = np.zeros([max_length - leng])
                    choice = np.append(choice, pads)
                choices.append(choice)
            return np.array(choices)


def load_spectrogram(filename, source_sentences_ids):
    data = pickle.load(open(filename, "rb"))
    new_data = []
    for id in source_sentences_ids:
        spectrogram = data[id].T
        new_data.append(spectrogram)
    return np.array(new_data)


def load_lexicons(path='person_lexicons'):
    roles = {'sheldon': 0, 'leonard': 1, 'penny': 2, 'howard': 3, 'raj': 4, 'amy': 5}
    lexicons = [[], [], [], [], [], []]
    for file in os.listdir(path):
        if '_id' not in file: continue
        person_name = file.split('_')[0]
        with open(os.path.join(path, file), encoding='utf-8') as f:
            lexicon_ids = f.readlines()
        lexicon = [int(lexicon_id) for lexicon_id in lexicon_ids]
        lexicons[roles[person_name]].extend(lexicon)
    return np.array(lexicons)


def load_embedding(path):
    ori_emb = []
    with open(path) as f:
        for line in f.readlines():
            ori_emb.append([float(fe) for fe in line.split()])
    ori_emb = np.array(ori_emb)

    ori_p_emb = []
    with open("model/p_emb.tsv") as f:
        for line in f.readlines():
            ori_p_emb.append([float(fe) for fe in line.split()])
    ori_p_emb = np.array(ori_p_emb)
    return ori_emb, ori_p_emb
