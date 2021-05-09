from utils.utils import init_embeddings, compute_loss, \
        loadfile, get_PEC_config, get_training_config, compute_test_perplexity, load, load_spectrogram, id2_word,\
        load_lexicons

import argparse
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
import os


def parse_args():
    '''
    Parse Emotional Chatting Machine arguments.
    '''
    parser = argparse.ArgumentParser(description="Run PEC inference.")

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
    embeddings = init_embeddings(vocab_size, embed_size, name=name)

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

    try:
        saved_global_step = load(saver, sess, logdir)
        if saved_global_step is None:
            raise ValueError("Cannot find the checkpoint to restore from.")

    except Exception:
        print("Something went wrong while restoring checkpoint. ")
        raise

    # ##### Inference #####
    # Load data
    print("Loading inference data ...")

    # id_0, id_1, id_2 preserved for SOS, EOS, constant zero padding
    embed_shift = 3

    lexicons = load_lexicons() + embed_shift

    test_source_sentences_ids, test_source_person, test_source_data = loadfile(test_s_filename, is_dialog=True,
                                                                               is_source=True,
                                                                               max_length=s_max_leng)
    test_source_data += embed_shift
    test_target_sentences_ids, test_target_person, test_target_data, test_category_data = loadfile(test_t_filename,
                                                                                                   is_dialog=True,
                                                                                                   is_source=False,
                                                                                                   max_length=t_max_leng)
    test_target_data += embed_shift
    test_spectrogram_data = load_spectrogram(sp_filename, test_source_sentences_ids)
    test_choice_data = loadfile(test_q_filename, is_dialog=False, is_source=False,
                                max_length=t_max_leng)
    test_choice_data[test_choice_data < 0] = 0
    test_choice_data = test_choice_data.astype(np.float32)

    test_masks = (test_target_data >= embed_shift)
    test_masks = np.append(
        np.ones([len(test_masks), 1], dtype=bool), test_masks, axis=1)
    test_masks = test_masks[:, :-1]
    print("\tDone.")

    # test
    print("testing")
    if test_source_data is not None:
        CE_words = N_words = 0.0
        for start in range(0, len(test_source_data), batch_size):
            test_feed_dict = {
                source_ids: test_source_data[start:start + batch_size],
                target_ids: test_target_data[start:start + batch_size],
                person_ids: test_target_person[start:start + batch_size],
                spectrogram: test_spectrogram_data[start:start + batch_size],
                choice_qs: test_choice_data[start:start + batch_size],
                emo_cat: test_category_data[start:start + batch_size],
                sequence_mask: test_masks[start:start + batch_size],
                lexicons_ids: lexicons,
                is_train: False,
            }
            CE_word, N_word = compute_test_perplexity(
                sess, CE, test_masks[start:start + batch_size], test_feed_dict)
            CE_words += CE_word
            N_words += N_word

        print("test_perp: {:.3f}".format(np.exp(CE_words / N_words)))

        infer_results = []
        for start in range(0, len(test_source_data), infer_batch_size):
            # infer_result = sess.run(infer_outputs,
            #                         feed_dict={source_ids: test_source_data[start:start + infer_batch_size],
            #                                    spectrogram: test_spectrogram_data[start:start + infer_batch_size],
            #                                    person_ids: test_target_person[start:start + infer_batch_size],
            #                                    emo_cat: test_category_data[start:start + infer_batch_size],
            #                                    lexicons_ids: lexicons,
            #                                    is_train: False,
            #                                    })
            #
            # infer_result = infer_result.ids[:, :, 0]
            # if infer_result.shape[1] < max_iter:
            #     l_pad = max_iter - infer_result.shape[1]
            #     infer_result = np.concatenate((infer_result, np.ones((infer_batch_size, l_pad))), axis=1)
            # else:
            #     infer_result = infer_result[:, :max_iter]
            tmp_result = []
            scores = []
            for i in range(num_emo):
                cat = i * np.ones([len(test_target_person[start:start + infer_batch_size])])
                infer_result, sco = sess.run([infer_outputs, score],
                                        feed_dict={source_ids: test_source_data[start:start + infer_batch_size],
                                                   spectrogram: test_spectrogram_data[start:start + infer_batch_size],
                                                   #spectrogram: np.zeros([len(test_source_data[start:start + infer_batch_size]), 400, 200]),
                                                   person_ids: test_target_person[start:start + infer_batch_size],
                                                   emo_cat: cat,
                                                   lexicons_ids: lexicons,
                                                   is_train: False,
                                                   })

                infer_result = infer_result.ids[:, :, 0]
                if infer_result.shape[1] < max_iter:
                    l_pad = max_iter - infer_result.shape[1]
                    infer_result = np.concatenate(
                        (infer_result, np.ones((infer_batch_size, l_pad))), axis=1)
                else:
                    infer_result = infer_result[:, :max_iter]
                tmp_result.append(infer_result)
                scores.append(sco)
            tmp_result = np.transpose(np.array(tmp_result), [1, 0, 2])
            scores = np.array(scores)
            scores = np.exp(scores)/np.sum(np.exp(scores), axis=0)
            scores = np.transpose(np.array(scores), [1, 0])
            scores[range(infer_batch_size), test_category_data[start:start + infer_batch_size]] += 1
            ind = np.argmax(scores, axis=-1)
            infer_results.extend(tmp_result[range(tmp_result.shape[0]), test_category_data[start:start + infer_batch_size]])
            #infer_results.extend(infer_result)

        final_result = np.array(infer_results) - embed_shift
        final_result[final_result >= vocab_size] -= (vocab_size + embed_shift)

        final_result = id2_word(final_result.astype(int).tolist())
        with open(os.path.join(test_output, "PEC_out_emo.tsv"), "w") as f:
            f.writelines('\n'.join(["0\t0\t" + str(emo) + "\t" + ' '.join(sen) for emo, sen in
                                    zip(test_category_data, final_result)]) + '\n')
        with open(os.path.join(test_output, "PEC_out_per.tsv"), "w") as f:
            f.writelines('\n'.join(["0\t0\t" + str(per) + "\t" + ' '.join(sen) for per, sen in
                                    zip(test_target_person, final_result)]) + '\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
