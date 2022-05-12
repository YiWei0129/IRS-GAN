
from __future__ import division
import numpy as np
import tensorflow as tf
import scipy.io as scio
from numpy.linalg import svd
import datetime
import sys




def model_inputs(M, N, z_dim_G ,z_dim_R):
    ## Real imag
    inputs_real = tf.placeholder(tf.float32, (None, M * N * 2), name='input_real')
    ## input z
    inputs_z_G = tf.placeholder(tf.float32, (None, z_dim_G), name='input_z_G')
    inputs_z_R = tf.placeholder(tf.float32, (None, z_dim_R), name='input_z_R')
    ## Learning rate
    learning_rate = tf.placeholder(tf.float32, name='lr')
    lr_b = tf.placeholder(tf.float32, name='lr_b')
    return inputs_real, inputs_z_G, inputs_z_R, learning_rate ,lr_b


def discriminator(channel ,M ,N, reuse=False):

    # TODO: Implement Function


    with tf.variable_scope('discriminator', reuse=reuse):
        alpha = 0.2
        input_re = tf.transpose(tf.reshape(channel[: ,0: M *N] ,(-1 ,N ,M)) ,perm=[0, 2, 1])
        input_re = tf.reshape(input_re ,(-1 ,M ,N ,1))
        input_im = tf.transpose(tf.reshape(channel[: , M *N: 2 * M *N] ,(-1 ,N ,M)) ,perm=[0, 2, 1])
        input_im = tf.reshape(input_im ,(-1 ,M ,N ,1))
        input= tf.concat([input_re, input_im], axis=3)

        conv2_g = tf.layers.conv2d(inputs=input, filters=128, kernel_size=5, padding='same')
        conv2_g = tf.nn.relu(conv2_g)
        conv3_g = tf.layers.conv2d(inputs=conv2_g, filters=64, kernel_size=3, padding='same')
        conv3_g = tf.nn.relu(conv3_g)
        conv4_g = tf.layers.conv2d(inputs=conv3_g, filters=1, kernel_size=3, padding='same')
        h4 = tf.contrib.layers.flatten(conv4_g)
        h4 = tf.layers.dense(h4, N, tf.nn.tanh)
        D_logit = tf.layers.dense(h4, units=1 ,name='D_logit')
        Dout = tf.nn.sigmoid(D_logit)
    return Dout, D_logit





def generator_apart(z_G, z_R, M, N, BiasVar =0.22 ,is_train=True):
    with tf.variable_scope('generator', reuse=not is_train):
        with tf.variable_scope('generator_Bias' ,reuse=not is_train):
            init_rand_G_R = np.random.normal(size=(M, N) ,scale = BiasVar)
            init_rand_G_I = np.random.normal(size=(M, N) ,scale = BiasVar)
            init_rand_R_R = np.random.normal(size = (1, N) ,scale = BiasVar)
            init_rand_R_I = np.random.normal(size = (1, N) ,scale = BiasVar)
            GN_Bias_re_ = tf.Variable(init_rand_G_R, dtype=tf.float32, name='GN_Bias_re_')
            GN_Bias_im_ = tf.Variable(init_rand_G_I, dtype=tf.float32, name='GN_Bias_im_')
            R_Bias_re_ = tf.Variable(init_rand_R_R, dtype=tf.float32, name='R_Bias_re_')
            R_Bias_im_ = tf.Variable(init_rand_R_I, dtype=tf.float32, name='R_Bias_im_')

        with tf.variable_scope('generator_GN', reuse=not is_train):
            G_g = tf.layers.dense(z_G, M * N * 2, tf.nn.tanh)
            G_g_2D = tf.reshape(G_g, (-1, M, N, 2))
            G_conv1_g = tf.layers.conv2d(inputs=G_g_2D, filters=256, kernel_size=5, padding='same')
            G_conv1_g = tf.nn.tanh(G_conv1_g)
            G_BN1_g = tf.layers.batch_normalization(G_conv1_g)
            G_conv2_g = tf.layers.conv2d(inputs=G_BN1_g, filters=128, kernel_size=3, padding='same')
            G_conv2_g = tf.nn.tanh(G_conv2_g)
            G_BN2_g = tf.layers.batch_normalization(G_conv2_g)
            G_conv3_g = tf.layers.conv2d(inputs=G_BN2_g, filters=64, kernel_size=3, padding='same')
            G_conv3_g = tf.nn.tanh(G_conv3_g)
            G_BN3_g = tf.layers.batch_normalization(G_conv3_g)
            G_conv4_g = tf.layers.conv2d(inputs=G_BN3_g, filters=2, kernel_size=3, padding='same')
            G_out_temp = tf.complex(G_conv4_g[: ,: ,:, 0], G_conv4_g[: ,: ,:, 1])
            G_out_temp1 =  tf.reshape(G_out_temp ,(-1 ,M ,N))
            G_out = G_out_temp1 + tf.complex(GN_Bias_re_ ,GN_Bias_im_)

        with tf.variable_scope('generator_RN', reuse=not is_train):
            R_fc1_g = tf.layers.dense(z_R, N * 2, tf.nn.tanh)
            R_fc2_g = tf.layers.dense(R_fc1_g, N * 5, tf.nn.tanh)
            R_out_temp = tf.layers.dense(R_fc2_g, N * 2)
            R_out = tf.complex(R_out_temp[:, 0: N], R_out_temp[:, N:N * 2]) + tf.complex(R_Bias_re_ ,R_Bias_im_)
            Gout_temp = tf.matmul(G_out, tf.matrix_diag(R_out))
            Gout_temp1 = tf.concat([tf.reshape(tf.transpose(tf.real(Gout_temp)), (-1, M * N)),
                                    tf.reshape(tf.transpose(tf.imag(Gout_temp)), (-1, M * N))], axis=1)
            Gout_temp1 = tf.layers.dense(Gout_temp1 , M * N *4 ,tf.nn.tanh)
            Gout= tf.layers.dense(Gout_temp1, M * N * 2, tf.nn.tanh, name='Gout')
    return Gout


def WGAN_GP_model_loss(channel_real, input_z_G, input_z_R, M, N, BiasVar ,batch_size = 50 ,LAMBDA =10 ,Mu = 10):

    g_model = generator_apart(input_z_G, input_z_R, M, N, BiasVar ,is_train=True)
    d_model_real, d_logits_real = discriminator(channel_real, M ,N ,reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, M ,N, reuse=True)



    d_loss = -tf.reduce_mean(d_logits_real) - tf.nn.relu( 1 -tf.reduce_mean(d_logits_fake))
    g_loss = tf.nn.relu( 1 -tf.reduce_mean(d_logits_fake)) + Mu *tf.abs \
        (tf.reduce_mean(tf.abs(channel_real) ) -tf.reduce_mean(tf.abs(g_model)))


    alpha = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    interpolates = alpha * channel_real + (1 - alpha) * g_model
    grad = tf.gradients(discriminator(interpolates,  M ,N ,reuse=True), [interpolates])[0]
    slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
    gp = tf.reduce_mean((slop - 1.) ** 2)
    d_loss += LAMBDA * gp


    return d_loss, g_loss, d_model_fake, d_model_real, g_model ,d_logits_real ,d_logits_fake


def model_opt_WGAN_GP(d_loss, g_loss, learning_rate ,lr_b ,k):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    if k == 0:
        g_vars = [var for var in t_vars if (var.name.startswith('generator/generator_GN')  or var.name.startswith('generator/generator_RN'))]
    else:
        g_vars = [var for var in t_vars if var.name.startswith('generator/generator_RN')]

    g_vars_Bias = [var for var in t_vars if var.name.startswith('generator/generator_Bias')]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
        g_train_opt_Bias = tf.train.AdamOptimizer(lr_b, beta1=0.5).minimize(g_loss, var_list=g_vars_Bias)
        g_train_opt_temp = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)
        g_train_opt = tf.group(g_train_opt_Bias, g_train_opt_temp)
    return d_train_opt, g_train_opt ,g_vars ,d_vars



def train(BiasVar ,RandomVar ,singluar_real ,epoch_count, plot_every, batch_size, z_dim_G ,z_dim_R, M, N, load_path, lr_d, lr_g ,lr_bias ,iterIn_d, iterIn_g, H_set ,mean_max, k, G_RN_learning_rate, G_FG_learning_rate ,Mu ,LAMBDA):
    losses = []

    input_real, input_z_G ,input_z_R, lr ,lr_b = model_inputs(M, N, z_dim_G ,z_dim_R)

    d_loss, g_loss, d_model_fake, d_model_real, g_model ,d_logits_real ,d_logits_fake = WGAN_GP_model_loss(input_real, input_z_G, input_z_R, M, N, BiasVar, batch_size, LAMBDA ,Mu)
    d_opt, g_opt, g_vars ,d_vars = model_opt_WGAN_GP(d_loss, g_loss, lr, lr_b, k)



    steps = 0


    if k> 0 and part == 1:
        t_vars = tf.trainable_variables()
        var_G_RN = [var for var in t_vars if var.name.startswith('generator/generator_RN')]
        g_opt1 = tf.train.RMSPropOptimizer(G_RN_learning_rate).minimize(g_loss, var_list=var_G_RN)
        var_G_FG = [var for var in t_vars if
                    (var.name.startswith('generator/generator_FN') or var.name.startswith('generator/generator_GN'))]
        g_opt2 = tf.train.RMSPropOptimizer(G_FG_learning_rate).minimize(g_loss, var_list=var_G_FG)
        g_opt_DIFF = tf.group(g_opt1, g_opt2)

    Length, Height = H_set.shape

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=1000)
    load_pretrain_model = True

    with tf.Session(config=config) as sess:
        print("Start training model for {}-th user".format(k + 1))
        if k == 0:
            sess.run(tf.global_variables_initializer())
        else:
            if k == 1:
                if part == 1:

                    sess.run(tf.global_variables_initializer())
                    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
                        ['generator/generator_GN', 'discriminator'])

                    moudle_file_k0 = "xxxxxx"
                    saver = tf.train.Saver(variables_to_restore, max_to_keep=100)
                    saver.restore(sess, moudle_file_k0)


                else:
                    saver = tf.train.Saver(max_to_keep=1000)
                    moudle_file_k0 = "xxxxxxx"
                    saver.restore(sess, moudle_file_k0)
            else:
                saver = tf.train.Saver(max_to_keep=1000)
                moudle_file_k0 = "xxxxxx"
                saver.restore(sess, moudle_file_k0)
        start = datetime.datetime.now()

        saver_out = tf.train.Saver(max_to_keep=1000)

        change_lr = 0
        singular_test_set = []
        singular_var_test_set = []
        for epoch_i in range(epoch_count + 1):
            if ((epoch_i + 1) * batch_size % Length == 0):
                batch_channel = H_set[(epoch_i * batch_size) % Length: Length, :]
            else:
                batch_channel = H_set[(epoch_i * batch_size) % Length:(epoch_i + 1) * batch_size % Length, :]

            steps += 1
            ## Run optimizer
            batch_z_G = np.random.normal(size=(batch_size, z_dim_G), scale=RandomVar)
            batch_z_R = np.random.normal(size=(batch_size, z_dim_R), scale=RandomVar)
            for d_idx in range(iterIn_d):
                sess.run(d_opt, feed_dict={input_real: batch_channel,
                                           input_z_G: batch_z_G,
                                           input_z_R: batch_z_R,
                                           lr: lr_d
                                           })

            for g_idx in range(iterIn_g):
                if part == 1 and k > 0:
                    sess.run(g_opt_DIFF, feed_dict={input_real: batch_channel,
                                                    input_z_G: batch_z_G,
                                                    input_z_R: batch_z_R
                                                    })
                else:
                    sess.run(g_opt, feed_dict={input_real: batch_channel,
                                               input_z_G: batch_z_G,
                                               input_z_R: batch_z_R,
                                               lr: lr_g,
                                               lr_b: lr_bias})

            if steps == 1:

                train_loss_d = d_loss.eval({input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_model_real = d_model_real.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_model_fake = d_model_fake.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_logits_real = d_logits_real.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_logits_fake = d_logits_fake.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_loss_g = g_loss.eval({input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})

                losses.append((train_loss_d, train_loss_g))

                singular_set = []
                for i in range(1000):
                    test_z_G = np.random.normal(size=(1, z_dim_G))
                    test_z_R = np.random.normal(size=(1, z_dim_R))
                    g_temp = sess.run(g_model, feed_dict={input_z_G: test_z_G, input_z_R: test_z_R})
                    g_H_temp2 = g_temp[:, 0:M * N] + 1j * g_temp[:, M * N:2 * M * N]

                    g_H = np.reshape(g_H_temp2, (M, N))
                    u, s, vh = np.linalg.svd(g_H)
                    s_mean = np.mean(s)
                    singular_set.append(s_mean)

                singular_test = np.mean(singular_set)
                singular_test_set.append(singular_test)

                print("step 1...".format(epoch_i + 1, epoch_count),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "train_d_logits_real: {:.4f}...".format(np.mean(train_d_logits_real)),
                      "train_d_logits_fake: {:.4f}...".format(np.mean(train_d_logits_fake)),
                      "train_d_model_real: {:.4f}...".format(np.mean(train_d_model_real)),
                      "train_d_model_fake: {:.4f}...".format(np.mean(train_d_model_fake)),
                      "Generator Loss: {:.4f}".format(train_loss_g))

            if steps % 100 == 0:
                train_loss_d = d_loss.eval({input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_model_real = d_model_real.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_model_fake = d_model_fake.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_logits_real = d_logits_real.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_d_logits_fake = d_logits_fake.eval(
                    {input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})
                train_loss_g = g_loss.eval({input_real: batch_channel, input_z_G: batch_z_G, input_z_R: batch_z_R})

                losses.append((train_loss_d, train_loss_g))

                print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "train_d_logits_real: {:.4f}...".format(np.mean(train_d_logits_real)),
                      "train_d_logits_fake: {:.4f}...".format(np.mean(train_d_logits_fake)),
                      "train_d_model_real: {:.4f}...".format(np.mean(train_d_model_real)),
                      "train_d_model_fake: {:.4f}...".format(np.mean(train_d_model_fake)),
                      "Generator Loss: {:.4f}".format(train_loss_g))

            if epoch_i % plot_every == 0:
                print("save model step_{}".format(epoch_i))
                saver_out.save(sess, load_path + str(epoch_i))

                singular_set = []
                for i in range(1000):
                    test_z_G = np.random.normal(size=(1, z_dim_G), scale=RandomVar)
                    test_z_R = np.random.normal(size=(1, z_dim_R), scale=RandomVar)
                    g_temp = sess.run(g_model, feed_dict={input_z_G: test_z_G, input_z_R: test_z_R})
                    g_H_temp2 = g_temp[:, 0:M * N] + 1j * g_temp[:, M * N:2 * M * N]
                    g_H = mean_max * np.transpose(np.reshape(g_H_temp2, (N, M)))
                    u, s, vh = np.linalg.svd(g_H)
                    s_mean = np.mean(s)
                    singular_set.append(s_mean)

                singular_test = np.mean(singular_set)
                singular_var_test = np.var(singular_set)
                singular_test_set.append(singular_test)
                singular_var_test_set.append(singular_var_test)

                if (abs(singular_test - singluar_real) <= 0.03) and (change_lr == 0):
                    change_lr = 1
                    lr_d = 0.1 * lr_d
                    lr_g = 0.1 * lr_g
                    lr_bias = 0.1 * lr_bias

                print(
                    "mean singular value test: {:4f}, var singular value test: {:4f}, lr_d = {:10f}, lr_g = {:10f}, lr_b = {:8f}".format(
                        singular_test, singular_var_test, lr_d, lr_g, lr_bias))

                if epoch_i > 0:
                    print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "train_d_logits_real: {:.4f}...".format(np.mean(train_d_model_real)),
                          "train_d_logits_fake: {:.4f}...".format(np.mean(train_d_model_fake)),
                          "Generator Loss: {:.4f}".format(train_loss_g),
                          "singular mean: {:.4f}".format(singular_test))

        end = datetime.datetime.now()
        print("The total training time is {}".format(end - start))
    return singular_test_set, singular_var_test_set


# Initial
batch_size = 50
beta1 = 0.5
beta2 = 0.9
epochs = 30000
plot_every = 1000
M = 6
N = 32
k = 1
z_dim_G = M * N
z_dim_R = N
testNum = 1000
# learning rate
lr_d = 0.0000001
lr_g = 0.0000001
lr_bias = 0
iterIn_d = 5
iterIn_g = 1
part = 1  # 是否固定G不动；
Mu = 10
LAMBDA = 10
G_RN_learning_rate = 0.0001
G_FG_learning_rate = 0.0001
RandomVar = 0.357
BiasVar = 0.36

# load training data
data_path = "xxxxx"
data = scio.loadmat(data_path)
H_set = data.get('H_set')
mean_max = data.get('mean_max')
load_path = "xxxxx"

singular_real_set = []
for i in range(10000):
    H_real_temp = H_set[i:i + 1, :]
    H_real_temp2 = H_real_temp[:, 0:M * N] + 1j * H_real_temp[:, M * N:2 * M * N]
    H_real = np.transpose(np.reshape(H_real_temp2, (N, M)))
    u, s_real, vh = np.linalg.svd(H_real)
    s_real_mean = np.mean(s_real)
    singular_real_set.append(s_real_mean)
singluar_real = np.mean(singular_real_set)


# train
start = datetime.datetime.now()
singular_test_set, singular_var_test_set = train(BiasVar, RandomVar, singluar_real, epochs, plot_every, batch_size,
                                                 z_dim_G, z_dim_R, M, N, load_path, lr_d, lr_g, lr_bias,
                                                 iterIn_d, iterIn_g, H_set / mean_max, mean_max, k, G_RN_learning_rate,
                                                 G_FG_learning_rate, Mu, LAMBDA)
end = datetime.datetime.now()
print("The total training time is {}".format(end - start))
















