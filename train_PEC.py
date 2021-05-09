from utils.utils import init_embeddings, compute_loss, compute_perplexity,\
        loadfile, get_PEC_config, get_training_config, compute_test_perplexity, load, save, load_spectrogram, \
        load_lexicons, load_embedding

import argparse
import time
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib
import os
from pprint import pprint
import copy

# matplotlib.use('agg')

# import matplotlib.pyplot as plt   # noqa


def parse_args():
    '''
    Parse Emotional Chatting Machine (PEC) arguments.
    '''
    parser = argparse.ArgumentParser(description="Run PEC training.")

    parser.add_argument('--config', nargs='?',
                        default='./config_PEC.yaml',
                        help='Configuration file for model specifications')

    return parser.parse_args()


def main(args):
    # loading configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)["configuration"]

    name = config["Name"]

    # Construct or load embeddings
    print("Initializing embeddings ...")
    vocab_size = config["embeddings"]["vocab_size"]
    embed_size = config["embeddings"]["embed_size"]
    per_num = config["embeddings"]["person_num"]
    per_embed_size = config["embeddings"]["person_embed_size"]
    ori_emb, ori_p_emb = load_embedding("model/emb.tsv")
    embeddings = init_embeddings(vocab_size, embed_size, initial_values=ori_emb, name=name)

    print("\tDone.")

    # Build the model and compute losses
    source_ids = tf.placeholder(tf.int32, [None, 40], name="source")
    target_ids = tf.placeholder(tf.int32, [None, 40], name="target")
    person_ids = tf.placeholder(tf.int32, [None], name="person_ids")
    lexicons_ids = tf.placeholder(tf.int32, [per_num, 1000], name="lexicons_ids")
    spectrogram = tf.placeholder(tf.float32, [None, 400, 200], name="audio")
    sequence_mask = tf.placeholder(tf.bool, [None, 40], name="mask")
    choice_qs = tf.placeholder(tf.float32, [None, 40], name="choice")
    emo_cat = tf.placeholder(tf.int32, [None], name="emotion_category")
    is_train = tf.placeholder(tf.bool)

    (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
     dec_num_layers, dec_num_units, dec_cell_type, state_pass,
     num_emo, emo_cat_units, emo_int_units,
     infer_batch_size, beam_size, max_iter,
     attn_num_units, l2_regularize, word_config,
     spectrogram_config, lstm_int_num, batch_size, loss_weight) = get_PEC_config(config)

    print("Building model architecture ...")
    CE, loss, cla_loss, train_outs, infer_outputs, score = compute_loss(
        source_ids, target_ids, sequence_mask, choice_qs, embeddings,
        enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
        dec_num_layers, dec_num_units, dec_cell_type, state_pass,
        num_emo, emo_cat, emo_cat_units, emo_int_units, infer_batch_size,
        spectrogram, word_config, per_num, person_ids, per_embed_size,
        spectrogram_config, loss_weight, lstm_int_num, is_train, False, lexicons_ids,
        beam_size, max_iter, attn_num_units, l2_regularize, name)
    print("\tDone.")

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    (logdir, restore_from, learning_rate, gpu_fraction, max_checkpoints,
     train_steps, batch_size, print_every, checkpoint_every, s_filename,
     t_filename, q_filename, s_max_leng, t_max_leng,
     dev_s_filename, dev_t_filename, dev_q_filename,
     loss_fig, perp_fig, sp_filename, sp_max_leng,
     test_s_filename, test_t_filename, test_q_filename, test_output) = get_training_config(config, "training")

    is_overwritten_training = logdir != restore_from

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       epsilon=1e-4)
    trainable = tf.trainable_variables()

    gradients = tf.gradients(loss, trainable)
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
    optim = optimizer.apply_gradients(zip(clipped_gradients, trainable))

    # optim = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=max_checkpoints)

    # BN
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except Exception:
        print("Something went wrong while restoring checkpoint. "
              "Training is terminated to avoid the overwriting.")
        raise

    # ##### Training #####
    # Load data
    print("Loading data ...")

    # id_0, id_1, id_2 preserved for SOS, EOS, constant zero padding
    embed_shift = 3

    lexicons = load_lexicons() + embed_shift

    source_sentences_ids, source_person, source_data = loadfile(s_filename, is_dialog=True, is_source=True,
                                                                max_length=s_max_leng)
    source_data += embed_shift
    target_sentences_ids, target_person, target_data, category_data = loadfile(t_filename, is_dialog=True, is_source=False,
                                                                               max_length=t_max_leng)
    target_data += embed_shift

    spectrogram_data = load_spectrogram(sp_filename, source_sentences_ids)
    choice_data = loadfile(q_filename, is_dialog=False, is_source=False, max_length=t_max_leng)
    choice_data = choice_data.astype(np.float32)

    masks = (target_data >= embed_shift)
    masks = np.append(np.ones([len(masks), 1], dtype=bool), masks, axis=1)
    masks = masks[:, :-1]
    n_data = len(source_data)

    dev_source_data = None
    if dev_s_filename is not None:
        dev_source_sentences_ids, dev_source_person, dev_source_data = loadfile(dev_s_filename, is_dialog=True, is_source=True,
                                                                                max_length=s_max_leng)
        dev_source_data += embed_shift
        dev_target_sentences_ids, dev_target_person, dev_target_data, dev_category_data = loadfile(dev_t_filename, is_dialog=True,
                                                                                                     is_source=False,
                                                                                                     max_length=t_max_leng)
        dev_target_data += embed_shift
        dev_spectrogram_data = load_spectrogram(sp_filename, dev_source_sentences_ids)
        dev_choice_data = loadfile(dev_q_filename, is_dialog=False, is_source=False,
                                   max_length=t_max_leng)
        dev_choice_data[dev_choice_data < 0] = 0
        dev_choice_data = dev_choice_data.astype(np.float32)

        dev_masks = (dev_target_data >= embed_shift)
        dev_masks = np.append(
            np.ones([len(dev_masks), 1], dtype=bool), dev_masks, axis=1)
        dev_masks = dev_masks[:, :-1]
    print("\tDone.")

    # Training
    last_saved_step = saved_global_step
    num_steps = saved_global_step + train_steps
    losses = []
    cla_losses = []
    steps = []
    perps = []
    dev_perps = []

    print("Start training ...")
    try:
        step = last_saved_step
        for step in range(saved_global_step + 1, num_steps):
            start_time = time.time()
            rand_indexes = np.random.choice(n_data, batch_size)
            source_batch = source_data[rand_indexes]
            target_batch = target_data[rand_indexes]
            person_batch = target_person[rand_indexes]
            spectrogram_batch = spectrogram_data[rand_indexes]
            mask_batch = masks[rand_indexes]
            choice_batch = choice_data[rand_indexes]
            emotions = category_data[rand_indexes]

            feed_dict = {
                source_ids: source_batch,
                target_ids: target_batch,
                person_ids: person_batch,
                spectrogram: spectrogram_batch,
                sequence_mask: mask_batch,
                choice_qs: choice_batch,
                emo_cat: emotions,
                lexicons_ids: lexicons,
                is_train: True,
            }
            loss_value, cla_value, _, __ = sess.run([loss, cla_loss, optim, extra_update_ops], feed_dict=feed_dict)
            losses.append(loss_value)
            cla_losses.append(cla_value)

            duration = time.time() - start_time

            if step % print_every == 0:
                # train perplexity
                t_perp = compute_perplexity(sess, CE, mask_batch, feed_dict)
                perps.append(t_perp)

                # dev perplexity
                dev_str = ""
                if dev_source_data is not None:
                    CE_words = N_words = 0.0
                    for start in range(0, len(dev_source_data), batch_size):
                        dev_feed_dict = {
                            source_ids: dev_source_data[start:start + batch_size],
                            target_ids: dev_target_data[start:start + batch_size],
                            person_ids: dev_target_person[start:start + batch_size],
                            spectrogram: dev_spectrogram_data[start:start + batch_size],
                            choice_qs: dev_choice_data[start:start + batch_size],
                            emo_cat: dev_category_data[start:start + batch_size],
                            sequence_mask: dev_masks[start:start + batch_size],
                            lexicons_ids: lexicons,
                            is_train: False,
                        }
                        CE_word, N_word = compute_test_perplexity(
                            sess, CE, dev_masks[start:start + batch_size], dev_feed_dict)
                        CE_words += CE_word
                        N_words += N_word

                    dev_str = "dev_prep: {:.3f}, ".format(np.exp(CE_words/N_words))
                    dev_perps.append(np.exp(CE_words/N_words))

                steps.append(step)
                info = 'step {:d}, loss = {:.6f}, cla_loss = {:.6f} '
                info += 'perp: {:.3f}, {}({:.3f} sec/step)'
                print(info.format(step, loss_value, cla_value, t_perp, dev_str, duration))

            if step % checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C so save message is on its own line.
        print()

    finally:
        if step > last_saved_step:
           save(saver, sess, logdir, step)

        # # plot loss
        # plt.figure()
        # plt.plot(losses)
        # plt.title("Total loss")
        # plt.xlabel("step")
        # plt.savefig(loss_fig)
        #
        # # plot loss
        # plt.figure()
        # plt.plot(cla_losses)
        # plt.title("Classification loss")
        # plt.xlabel("step")
        # plt.savefig("Classification loss")
        #
        # # plot perplexity
        # plt.figure()
        # if len(perps) > len(steps):
        #     perps.pop()
        # plt.plot(steps[5:], perps[5:], label="train")
        # if dev_source_data is not None:
        #     plt.plot(steps[5:], dev_perps[5:], label="dev")
        # plt.title("Perplexity")
        # plt.xlabel("step")
        # plt.legend()
        # plt.savefig(perp_fig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
