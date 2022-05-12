from __future__ import division
import numpy as np
import tensorflow as tf
import scipy.io as scio
from numpy.linalg import svd
import matplotlib.pyplot as plt
import datetime
import sys


def Jmodel_inputs(M, N, K, z_dim_G, z_dim_R):
    ## Real imag
    inputs_real_set = tf.placeholder(tf.float32, (None, M * N * 2 * K), name='input_real_set')
    ## input z
    inputs_z_G = tf.placeholder(tf.float32, (None, z_dim_G), name='input_z_G')
    inputs_z_R = tf.placeholder(tf.float32, (None, z_dim_R * K), name='input_z_R')
    ## Learning rate
    learning_rate = tf.placeholder(tf.float32, name='lr')
    lr_b_G = tf.placeholder(tf.float32, name='lr_b')
    lr_b_R_set = tf.placeholder(tf.float32, K, name='lr_b')
    return inputs_real_set, inputs_z_G, inputs_z_R, learning_rate, lr_b_G, lr_b_R_set



def discriminator_Multi(channel_set, M, N, K, reuse=False):
    D_logit_set = []
    Dout_set = []
    for k in range(1, K + 1):
        with tf.variable_scope('discriminator' + str(k), reuse=reuse):
            alpha = 0.2
            channel = channel_set[:, (k - 1) * M * N * 2:k * M * N * 2]
            input_re = tf.transpose(tf.reshape(channel[:, 0:M * N], (-1, N, M)), perm=[0, 2, 1])
            input_re = tf.reshape(input_re, (-1, M, N, 1))
            input_im = tf.transpose(tf.reshape(channel[:, M * N:2 * M * N], (-1, N, M)), perm=[0, 2, 1])
            input_im = tf.reshape(input_im, (-1, M, N, 1))
            input = tf.concat([input_re, input_im], axis=3)
            conv2_g = tf.layers.conv2d(inputs=input, filters=128, kernel_size=5, padding='same')
            conv2_g = tf.nn.relu(conv2_g)
            conv3_g = tf.layers.conv2d(inputs=conv2_g, filters=64, kernel_size=3, padding='same')
            conv3_g = tf.nn.relu(conv3_g)
            conv4_g = tf.layers.conv2d(inputs=conv3_g, filters=1, kernel_size=3, padding='same')
            h4 = tf.contrib.layers.flatten(conv4_g)
            h4 = tf.layers.dense(h4, N, tf.nn.tanh)
            D_logit = tf.layers.dense(h4, units=1, name='D_logit_' + str(k))

            Dout = tf.nn.sigmoid(D_logit, name='D_out_' + str(k))
            D_logit_set.append(D_logit)
            Dout_set.append(Dout)
    return Dout_set, D_logit_set



def generator_Japart(z_G, z_R, z_R_dim, M, N, K, BiasVar_G, BiasVar_R_Set, is_train=True):
    # Generate network fot trflected
    with tf.variable_scope('generator', reuse=not is_train):
        with tf.variable_scope('generator_Bias_G', reuse=not is_train):
            init_rand_G_R = np.random.normal(size=(M, N), scale=BiasVar_G)
            init_rand_G_I = np.random.normal(size=(M, N), scale=BiasVar_G)
            GN_Bias_re_ = tf.Variable(init_rand_G_R, dtype=tf.float32, name='GN_Bias_re_')
            GN_Bias_im_ = tf.Variable(init_rand_G_I, dtype=tf.float32, name='GN_Bias_im_')

        R_Bias_re_set = []
        R_Bias_im_set = []
        for k in range(K):
            with tf.variable_scope('generator_Bias_R_' + str(k), reuse=not is_train):
                init_rand_R_R = np.random.normal(size=(1, N), scale=BiasVar_R_Set[k])
                init_rand_R_I = np.random.normal(size=(1, N), scale=BiasVar_R_Set[k])
                R_Bias_re_ = tf.Variable(init_rand_R_R, dtype=tf.float32, name='R_Bias_re_' + str(k))
                R_Bias_im_ = tf.Variable(init_rand_R_I, dtype=tf.float32, name='R_Bias_im_' + str(k))
                R_Bias_re_set.append(R_Bias_re_)
                R_Bias_im_set.append(R_Bias_im_)

        with tf.variable_scope('generator_GN', reuse=not is_train):

            G_g = tf.layers.dense(z_G, M * N * 2, tf.nn.tanh)
            G_g_2D = tf.reshape(G_g, (-1, M, N, 2))
            G_conv1_g = tf.layers.conv2d(inputs=G_g_2D, filters=256, kernel_size=5, padding='same')
            G_conv1_g = tf.nn.tanh(G_conv1_g)

            G_conv2_g = tf.layers.conv2d(inputs=G_conv1_g, filters=128, kernel_size=3, padding='same')
            G_conv2_g = tf.nn.tanh(G_conv2_g)

            G_conv3_g = tf.layers.conv2d(inputs=G_conv2_g, filters=64, kernel_size=3, padding='same')
            G_conv3_g = tf.nn.tanh(G_conv3_g)

            G_conv4_g = tf.layers.conv2d(inputs=G_conv3_g, filters=2, kernel_size=3, padding='same')
            G_out_temp = tf.complex(G_conv4_g[:, :, :, 0], G_conv4_g[:, :, :, 1])
            G_out_temp1 = tf.reshape(G_out_temp, (-1, M, N))
            G_out = G_out_temp1 + tf.complex(GN_Bias_re_, GN_Bias_im_)

        with tf.variable_scope('generator_RN', reuse=not is_train):
            for k in range(1, K + 1):
                R_fc1_g = tf.layers.dense(z_R[:, z_R_dim * (k - 1):z_R_dim * k], N * 2, tf.nn.tanh)
                R_fc2_g = tf.layers.dense(R_fc1_g, N * 5, tf.nn.tanh)
                R_out_temp = tf.layers.dense(R_fc2_g, N * 2)

                R_out = tf.complex(R_out_temp[:, 0: N], R_out_temp[:, N:N * 2]) + tf.complex(R_Bias_re_set[k - 1],
                                                                                             R_Bias_im_set[k - 1])

                Gout_temp = tf.matmul(G_out, tf.matrix_diag(R_out))
                Gout_temp1 = tf.concat([tf.reshape(tf.transpose(tf.real(Gout_temp)), (-1, M * N)),
                                        tf.reshape(tf.transpose(tf.imag(Gout_temp)), (-1, M * N))], axis=1)
                Gout_temp1 = tf.layers.dense(Gout_temp1, M * N * 4, tf.nn.tanh)
                Gout = tf.layers.dense(Gout_temp1, M * N * 2, tf.nn.tanh, name='Gout_' + str(k))
                if k == 1:
                    Gout_set = Gout
                else:
                    Gout_set = tf.concat([Gout_set, Gout], axis=1, name='Gout_set')
    return Gout_set


def JWGAN_GP_model_loss(H_mean_set, channel_real, input_z_G, input_z_R, z_R_dim, M, N, K, BiasVar_G, BiasVar_R_Set,
                        batch_size=50, LAMBDA=10, Mu=10):
    g_model = generator_Japart(input_z_G, input_z_R, z_R_dim, M, N, K, BiasVar_G, BiasVar_R_Set, is_train=True)

    channel_real_set = []
    g_model_set = []
    for k in range(K):
        channel_real_set.append(channel_real[:, 2 * M * N * k:2 * M * N * (k + 1)])
        g_model_set.append(g_model[:, 2 * M * N * k:2 * M * N * (k + 1)])
    d_model_real_set, d_logits_real_set = discriminator_Multi(channel_real, M, N, K, reuse=False)

    d_model_fake_set, d_logits_fake_set = discriminator_Multi(g_model, M, N, K, reuse=True)

    d_loss_set = []
    Jg_loss = 0

    alpha = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    interpolates = alpha * channel_real + (1 - alpha) * g_model
    Dout_set, D_logit_set = discriminator_Multi(interpolates, M, N, K, reuse=True)

    for k in range(K):
        one_v = np.ones(shape=[batch_size, 1])
        d_loss = -tf.reduce_mean(d_logits_real_set[k]) - tf.reduce_mean(
            tf.nn.relu(one_v - d_logits_fake_set[k]))
        Jg_loss += tf.reduce_mean(tf.nn.relu(one_v - d_logits_fake_set[k])) + Mu[k] * tf.reduce_mean(
            tf.abs(H_mean_set[k] - tf.reduce_mean(g_model_set[k], axis=0)))

        grad = tf.gradients(Dout_set[k], [interpolates])[0]
        slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
        gp = tf.reduce_mean((slop - 1.) ** 2)
        d_loss += LAMBDA * gp
        d_loss_set.append(d_loss)

    return d_loss_set, Jg_loss, d_model_fake_set, d_model_real_set, g_model_set, d_logits_real_set, d_logits_fake_set


def model_opt_JWGAN_GP(d_loss_set, Jg_loss, learning_rate, lr_b_G, lr_b_R_set, K):
    t_vars = tf.trainable_variables()
    d_vars_set = []
    for k in range(K):
        d_vars = [var for var in t_vars if var.name.startswith('discriminator' + str(k + 1))]
        d_vars_set.append(d_vars)

    g_vars = [var for var in t_vars if
              (var.name.startswith('generator/generator_GN') or var.name.startswith('generator/generator_RN'))]

    g_vars_Bias_G = [var for var in t_vars if var.name.startswith('generator/generator_Bias_G')]
    g_vars_Bias_R_set = []
    for k in range(K):
        g_vars_Bias_R = [var for var in t_vars if var.name.startswith('generator/generator_Bias_R_' + str(k))]
        g_vars_Bias_R_set.append(g_vars_Bias_R)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    d_train_opt_set = []
    with tf.control_dependencies(update_ops):
        g_train_opt_Bias = tf.train.AdamOptimizer(lr_b_G, beta1=0.5).minimize(Jg_loss, var_list=g_vars_Bias_G)
        for k in range(K):
            g_train_opt_Bias_R = tf.train.AdamOptimizer(lr_b_R_set[k], beta1=0.5).minimize(Jg_loss,
                                                                                           var_list=g_vars_Bias_R_set[
                                                                                               k])
            g_train_opt_Bias = tf.group(g_train_opt_Bias, g_train_opt_Bias_R)
        g_train_opt_temp = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(Jg_loss, var_list=g_vars)
        Jg_train_opt = tf.group(g_train_opt_Bias, g_train_opt_temp)
        for k in range(K):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_loss_set[k],
                                                                                    var_list=d_vars_set[k])
            d_train_opt_set.append(d_train_opt)
    return d_train_opt_set, Jg_train_opt, g_vars, d_vars_set


def Jtrain(H_mean_set,  BiasVar_G, BiasVar_R_Set, RandomVar_G, RandomVar_R,
           epoch_count, batch_size, z_dim_G, z_dim_R, M, N, load_path, lr_d, lr_g, lr_bias_G, lr_bias_R_set,
           iterIn_d, iterIn_g, H_set_K, mean_max_set, K,
           Mu, LAMBDA):

    input_real_set, input_z_G, input_z_R, lr, lr_b_G, lr_b_R_set = Jmodel_inputs(M, N, K, z_dim_G, z_dim_R)

    d_loss_set, Jg_loss, d_model_fake_set, d_model_real_set, g_model_set, d_logits_real_set, d_logits_fake_set = JWGAN_GP_model_loss(
        H_mean_set, input_real_set, input_z_G, input_z_R, z_dim_R, M, N, K, BiasVar_G, BiasVar_R_Set, batch_size,
        LAMBDA, Mu)
    d_opt_set, Jg_opt, g_vars, d_vars_set = model_opt_JWGAN_GP(d_loss_set, Jg_loss, lr, lr_b_G, lr_b_R_set, K)

    steps = 0

    Length, Height = H_set.shape

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        print("Start training model for {} users".format(K))

        sess.run(tf.global_variables_initializer())

        start = datetime.datetime.now()

        saver_out = tf.train.Saver(max_to_keep=1000)


        singular_test_set = [[] for i in range(K)]
        singular_var_test_set = [[] for i in range(K)]
        for epoch_i in range(epoch_count + 1):
            if ((epoch_i + 1) * batch_size % Length == 0):
                batch_channel = H_set_K[(epoch_i * batch_size) % Length: Length, :]
            else:
                batch_channel = H_set_K[(epoch_i * batch_size) % Length:(epoch_i + 1) * batch_size % Length, :]
            steps += 1

            batch_z_G = np.random.normal(size=(batch_size, z_dim_G), scale=RandomVar_G)
            for k in range(K):
                if k == 0:
                    batch_z_R = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar_R[k])
                else:
                    batch_z_R_k = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar_R[k])
                    batch_z_R = np.concatenate([batch_z_R, batch_z_R_k], axis=1)

            if steps == 1:
                train_loss_g = Jg_loss.eval({input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                print("step {}/{}...".format(epoch_i + 1, epoch_count), "Generator Loss: {:.4f}".format(train_loss_g))
                for k in range(K):
                    train_loss_d = d_loss_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_model_real = d_model_real_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_model_fake = d_model_fake_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_logits_real = d_logits_real_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_logits_fake = d_logits_fake_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    print("user {}".format(k),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "train_d_logits_real: {:.4f}...".format(np.mean(train_d_logits_real)),
                          "train_d_logits_fake: {:.4f}...".format(np.mean(train_d_logits_fake)),
                          "train_d_model_real: {:.4f}...".format(np.mean(train_d_model_real)),
                          "train_d_model_fake: {:.4f}...".format(np.mean(train_d_model_fake)))
            print("save model JIRS_GAN_model_step_{}".format(epoch_i))
            saver_out.save(sess, load_path + str(epoch_i))

            singular_set = [[] for i in range(K)]
            H_mean_set_k = [[] for i in range(K)]
            for i in range(1000):
                batch_z_G = np.random.normal(size=(1, z_dim_G), scale=RandomVar_G)
                for k in range(K):
                    if k == 0:
                        batch_z_R = np.random.normal(size=(1, z_dim_R), scale=RandomVar_R[k])
                    else:
                        batch_z_R_k = np.random.normal(size=(1, z_dim_R), scale=RandomVar_R[k])
                        batch_z_R = np.concatenate([batch_z_R, batch_z_R_k], axis=1)
                g_temp = sess.run(g_model_set, feed_dict={input_z_G: batch_z_G, input_z_R: batch_z_R})

                for k in range(K):
                    g_H_temp2 = g_temp[k][:, 0:M * N] + 1j * g_temp[k][:, M * N:2 * M * N]
                    g_H = mean_max_set[k] * np.transpose(np.reshape(g_H_temp2, (N, M)))
                    H_mean_set_k[k].append(g_temp[k])
                    u, s, vh = np.linalg.svd(g_H)
                    s_mean = np.mean(s)
                    singular_set[k].append(s_mean)
                    for k in range(K):
                        singular_test = np.mean(singular_set[k])
                    singular_var_test = np.var(singular_set[k])
                    singular_test_set[k].append(singular_test)
                    singular_var_test_set[k].append(singular_var_test)
                    H_mean2_k = np.mean(H_mean_set_k[k], axis=0)
                    H_mean_loss = sum(sum(np.abs(H_mean2_k - H_mean_set[k])))

                    print(
                        "k = {}, H_mean_loss: {}, mean singular value test: {:4f}, var singular value test: {:4f}, lr_d = {:10f}, lr_g = {:10f}, lr_b_G = {:8f}, lr_b_R = {:8f}".format(
                            k, H_mean_loss,
                            singular_test, singular_var_test, lr_d, lr_g, lr_bias_G, lr_bias_R_set[k]))

                    for d_idx in range(iterIn_d):
                        batch_z_G = np.random.normal(size=(batch_size, z_dim_G), scale=RandomVar_G)
                    for k in range(K):
                        if k == 0:
                           batch_z_R = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar_R[k])
                        else:
                           batch_z_R_k = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar_R[k])
                           batch_z_R = np.concatenate([batch_z_R, batch_z_R_k], axis=1)

                    for k in range(K):
                        sess.run(d_opt_set[k], feed_dict={input_real_set: batch_channel,
                                                          input_z_G: batch_z_G,
                                                          input_z_R: batch_z_R,
                                                          lr: lr_d
                                                          })

                    if (d_idx + 1) % 200 == 0:
                        print("Epoch {}/{}...".format(epoch_i, epoch_count),
                              "d_iter {}/{}...".format(d_idx + 1, iterIn_d))
                    for k in range(K):
                        train_loss_d = d_loss_set[k].eval(
                            {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_model_real = d_model_real_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_model_fake = d_model_fake_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_logits_real = d_logits_real_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_logits_fake = d_logits_fake_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    print("user {}".format(k),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "train_d_logits_real: {:.4f}...".format(np.mean(train_d_logits_real)),
                          "train_d_logits_fake: {:.4f}...".format(np.mean(train_d_logits_fake)),
                          "train_d_model_real: {:.4f}...".format(np.mean(train_d_model_real)),
                          "train_d_model_fake: {:.4f}...".format(np.mean(train_d_model_fake)))

                    for g_idx in range(iterIn_g):
                        batch_z_G = np.random.normal(size=(batch_size, z_dim_G), scale=RandomVar_G)
                    for k in range(K):
                        if k == 0:
                             batch_z_R = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar_R[k])
                        else:
                             batch_z_R_k = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar_R[k])
                             batch_z_R = np.concatenate([batch_z_R, batch_z_R_k], axis=1)

                    sess.run(Jg_opt, feed_dict={input_real_set: batch_channel,
                                                input_z_G: batch_z_G,
                                                input_z_R: batch_z_R,
                                                lr: lr_g,
                                                lr_b_G: lr_bias_G,
                                                lr_b_R_set: lr_bias_R_set
                                                })

                    if (g_idx + 1) % 200 == 0:
                        train_loss_g = Jg_loss.eval(
                            {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    print("Epoch {}/{}...".format(epoch_i, epoch_count), "g_idx {}/{}...".format(g_idx + 1, iterIn_g),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    for k in range(K):
                        train_d_model_real = d_model_real_set[k].eval(
                            {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_model_fake = d_model_fake_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_logits_real = d_logits_real_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    train_d_logits_fake = d_logits_fake_set[k].eval(
                        {input_real_set: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                    print("user {}".format(k),
                          "train_d_logits_real: {:.4f}...".format(np.mean(train_d_logits_real)),
                          "train_d_logits_fake: {:.4f}...".format(np.mean(train_d_logits_fake)),
                          "train_d_model_real: {:.4f}...".format(np.mean(train_d_model_real)),
                          "train_d_model_fake: {:.4f}...".format(np.mean(train_d_model_fake)))

                    if steps >= 1:
                        print("save model JIRS_GAN_model_step_{}".format(epoch_i + 1), file=mylog)
                    print("save model JIRS_GAN_model_step_{}".format(epoch_i + 1))
                    saver_out.save(sess, load_path + str(epoch_i + 1))

                    singular_set = [[] for i in range(K)]
                    H_mean_set_k = [[] for i in range(K)]
                    for i in range(1000):
                        batch_z_G = np.random.normal(size=(1, z_dim_G), scale=RandomVar_G)
                    for k in range(K):
                        if k == 0:
                            batch_z_R = np.random.normal(size=(1, z_dim_R), scale=RandomVar_R[k])
                        else:
                            batch_z_R_k = np.random.normal(size=(1, z_dim_R), scale=RandomVar_R[k])
                            batch_z_R = np.concatenate([batch_z_R, batch_z_R_k], axis=1)


                    g_temp = sess.run(g_model_set, feed_dict={input_z_G: batch_z_G, input_z_R: batch_z_R})

                    for k in range(K):
                        g_H_temp2 = g_temp[k][:, 0:M * N] + 1j * g_temp[k][:, M * N:2 * M * N]
                    g_H = mean_max_set[k] * np.transpose(np.reshape(g_H_temp2, (N, M)))
                    H_mean_set_k[k].append(g_temp[k])

                    u, s, vh = np.linalg.svd(g_H)
                    s_mean = np.mean(s)
                    singular_set[k].append(s_mean)
                    for k in range(K):
                        singular_test = np.mean(singular_set[k])
                    singular_var_test = np.var(singular_set[k])
                    singular_test_set[k].append(singular_test)
                    singular_var_test_set[k].append(singular_var_test)
                    H_mean2_k = np.mean(H_mean_set_k[k], axis=0)
                    H_mean_loss = sum(sum(np.abs(H_mean2_k - H_mean_set[k])))

                    print(
                        "k = {}, H_mean_loss: {}, mean singular value test: {:4f}, var singular value test: {:4f}, lr_d = {:10f}, lr_g = {:10f}, lr_b_G = {:8f}, lr_b_R = {:8f}".format(
                            k, H_mean_loss,
                            singular_test, singular_var_test, lr_d, lr_g, lr_bias_G, lr_bias_R_set[k]))

                    end = datetime.datetime.now()
                    print("The total training time is {}".format(end - start))
    return singular_test_set, singular_var_test_set


# Initial
batch_size = 50
beta1 = 0.5
beta2 = 0.9
epochs = 50
plot_every = 1
M = 6
N = 32
K = 3
z_dim_G = M * N
z_dim_R = N
testNum = 10000
# learning rate
lr_d = 0.0000001
lr_g = 0.0000001
lr_bias_G = 0.0001
lr_bias_R_set = [0.0001, 0.0001, 0.0001]
iterIn_d = 1000
iterIn_g = 1000
# Generate structure
G_RN_learning_rate = 0.0001
Mu = 10
LAMBDA = 10

RandomVar_G = 0.8
RandomVar_R = [0.7, 0.904, 0.17]
BiasVar_G = 0.3
BiasVar_R_set = [0.55, 0.23, 0.12]

'''multiuser'''
data_path = "xxxxxxx"
data = scio.loadmat(data_path)
H_set_K = []
mean_max_set = []
load_path_set = []
for k in range(K):
    H_set = data.get('H_set{}'.format(k + 1))
    mean_max = data.get('mean_max{}'.format(k + 1))
    H_set_K.append(H_set)
    H_set_nor = H_set / mean_max
    mean_max_set.append(mean_max)
    if k == 0:
        H_SET_K_MAT = H_set_nor
    else:
        H_SET_K_MAT = np.concatenate([H_SET_K_MAT, H_set_nor], axis=1)

load_path = "xxxxxxxx"


H_mean_set = []
H_mean_mat_set = []
for k in range(K):
    H_set = H_set_K[k]
    singular_real_set = []
    H_mean = []
    H_mean_mat = []
    for i in range(10000):
        H_real_temp = H_set[i:i + 1, :]
        H_real_temp2 = H_real_temp[:, 0:M * N] + 1j * H_real_temp[:, M * N:2 * M * N]
        H_real = np.transpose(np.reshape(H_real_temp2, (N, M)))
        H_mean.append(H_real_temp)
        H_mean_mat.append(H_real)
        u, s_real, vh = np.linalg.svd(H_real)
        s_real_mean = np.mean(s_real)
        singular_real_set.append(s_real_mean)

    H_mean_k = np.mean(H_mean, axis=0)
    H_mean_k_mat = np.mean(H_mean_mat, axis=0)

    singluar_real = np.mean(singular_real_set)
    singular_var = np.var(singular_real_set)
    print("user{} singluar_real_mean = {:8f}, singular_real_var = {:8f} ".format(k, singluar_real, singular_var))
    H_mean_set.append(H_mean_k)

# train

singular_test_set, singular_var_test_set = Jtrain(H_mean_set, BiasVar_G,
                                                  BiasVar_R_set, RandomVar_G, RandomVar_R, epochs, plot_every,
                                                  batch_size, z_dim_G, z_dim_R, M, N, load_path, lr_d, lr_g, lr_bias_G,
                                                  lr_bias_R_set, iterIn_d, iterIn_g, H_SET_K_MAT, mean_max_set, K, Mu,
                                                  LAMBDA)












