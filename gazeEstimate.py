# Mukul Surajiwale
# ECSE 4965: Deep Learning
# Final Project
# 5/9/2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
from functools import partial
import matplotlib.image as mpimg
import pickle

# ------------------------------
# Loading and Preprocessing Data
# ------------------------------

# Import training data
npzfile = np.load("train_and_val.npz")
train_eye_left = npzfile["train_eye_left"]  # (48,000, 64, 64, 3)
train_y = npzfile["train_y"]                # (48,000, 2)

# Import validation data
val_eye_left = npzfile["val_eye_left"]
val_y = npzfile["val_y"]

print "Done importing "

# ----------------------------
# Utility Functions
# ----------------------------

# Utility function to normalize the data by divding by 255 and subtracting the mean
def normalizeData(train_eye_left, train_y, val_eye_left, val_y):
    print "Normalizing Data"
    # Convert to float32
    train_eye_left = np.array(train_eye_left, dtype="float32")
    train_y = np.array(train_y, dtype="float32")

    val_eye_left = np.array(val_eye_left, dtype="float32")
    val_y = np.array(val_y, dtype="float32")

    # Scaling
    train_eye_left = train_eye_left / 255.0
    val_eye_left = val_eye_left / 255.0

    # Mean normalization
    train_eye_left = train_eye_left - np.mean(train_eye_left)
    val_eye_left = val_eye_left - np.mean(val_eye_left)

    print "Data Normalization Complete"

    return train_eye_left, train_y, val_eye_left, val_y

# Utility function to perform a 2d convolution with a stride of 1
def conv2d(X, W, B):
    conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.relu(conv) + B

# Utility function to perform 2d max pooling with filter size of 2x2 and stride of 2x2
def maxPool(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# ----------------------------
# Model
# ----------------------------

batch_size = 200
graph = tf.Graph()

with graph.as_default():
    # Placeholders for all the data inputs
    eye_left = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name="Eye_left")
    eye_right = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name="Eye_right")
    face = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name="Face")
    face_mask = tf.placeholder(tf.float32, shape=(batch_size, 25, 25), name="Face_mask")
    Y = tf.placeholder(tf.float32, shape=(batch_size, 2), name="Expected_Output")

    # Placeholders for the validations inputs
    X_val = tf.placeholder(tf.float32, shape=(5000, 64, 64, 3))
    Y_val = tf.placeholder(tf.float32, shape=(5000, 2))

    weights = {
                "c1w": tf.get_variable(name="Conv1_Weights", shape=[7, 7, 3, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)),
                "c2w": tf.get_variable(name="Conv2_Weights", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)),
                "c3w": tf.get_variable(name="Conv3_Weights", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)),
                "wOut": tf.get_variable(name="Output_Weights", shape=[32768, 2], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    }

    biases = {
                "b1": tf.get_variable(name="Conv1_Bias", shape=[32], initializer=tf.constant_initializer(0.0)),
                "b2": tf.get_variable(name="Conv2_Bias", shape=[64], initializer=tf.constant_initializer(0.0)),
                "b3": tf.get_variable(name="Conv3_Bias", shape=[128], initializer=tf.constant_initializer(0.0)),
                "bOut": tf.get_variable(name="Output_Bias", shape=[2], initializer=tf.constant_initializer(0.0))
    }

    # CNN model with 3 convolution layers and 2 max pooling layers
    def cnnModel(inputData):
        with tf.name_scope("Conv1"):
            conv1 = conv2d(inputData, weights["c1w"], biases["b1"])
            pool1 = maxPool(conv1)

        with tf.name_scope("Conv2"):
            conv2 = conv2d(pool1, weights["c2w"], biases["b2"])
            pool2 = maxPool(conv2)

        with tf.name_scope("Conv3"):
            conv3 = conv2d(pool2, weights["c3w"], biases["b3"])

        with tf.name_scope("Fully_Connected"):
            shape = conv3.get_shape().as_list()
            finalShape = [shape[0], shape[1] * shape[2] * shape[3]]
            reshape = tf.reshape(conv3, finalShape)
            out = tf.matmul(reshape, weights["wOut"]) + biases["bOut"]

        return out

    # Training Prediction Operation
    with tf.name_scope("Predict_Op"):
        predict_op = cnnModel(eye_left)

    # Validation Prediction Operation
    with tf.name_scope("Val_Predict_Op"):
        val_pred = cnnModel(X_val)

    # Calulate Loss Values
    with tf.name_scope("Training_Loss"):
        loss = tf.reduce_mean(tf.square(predict_op - Y))

    # Training Error Computation
    with tf.name_scope("Training_Error"):
        training_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(predict_op - Y, 2), axis=1)))

    # Validation Error Computation
    with tf.name_scope("Validation_Error"):
        validation_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(val_pred - Y_val, 2), axis=1)))

    # Optimizer
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


    # Summaries to visualize the data in Tensorboard
    tf.summary.scalar("Loss", loss)
    tf.summary.scalar("Training_Error", training_error)
    tf.summary.image("Left_Eye_Input", eye_left, 10)

    # Save operations to collection
    tf.get_collection("validation_nodes")
    tf.add_to_collection("validation_nodes", eye_left)
    tf.add_to_collection("validation_nodes", eye_right)
    tf.add_to_collection("validation_nodes", face)
    tf.add_to_collection("validation_nodes", face_mask)
    tf.add_to_collection("validation_nodes", predict_op)

# ----------------------------
# Training
# ----------------------------

def trainNetwork(iterations, train_eye_left, train_y, val_eye_left, val_y):

    # Variables to store values to be visualized
    loss_values = []
    training_error_values = []
    validataion_error_values = []
    iters = []

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print "Variables Initialized"

        # Visualize the model in Tensorboard
        writer = tf.summary.FileWriter("/tmp/deepGaze/22")
        writer.add_graph(session.graph)

        # Visualize Summaries in Tensorboard
        merged_summary = tf.summary.merge_all()

        for i in range(iterations):
            # Select random batch
            offset = (i * batch_size) % (train_y.shape[0] - batch_size)
            batch_input = train_eye_left[offset:(offset + batch_size), :, :, :]
            batch_output = train_y[offset:(offset + batch_size)]

            train_dict = {eye_left: batch_input, Y: batch_output}
            _, l, t_error = session.run([optimizer, loss, training_error], feed_dict=train_dict)

            # Log information for Tensorboard
            s = session.run(merged_summary, feed_dict=train_dict)
            writer.add_summary(s, i)
            print i
            # Calculate training and validation error every epoch
            if (i % 240 == 0):
                # Calculate validation error
                v_error = session.run(validation_error, feed_dict={X_val: val_eye_left, Y_val: val_y})

                # Store the loss values, training error, and validation error
                iters.append(i)
                loss_values.append(l)
                training_error_values.append(t_error)
                validataion_error_values.append(v_error)

                print "------Step %d------" % (i+1)
                print "Training Loss: %f" % (l)
                print "Training Error: %f" % (t_error)
                print "Validation Error: %f" % (v_error)
                print "-------------------"

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(session, "my_model")

    return loss_values, training_error_values, validataion_error_values, iters

# ----------------------------
# Visualize
# ----------------------------

def visualizeTraining(loss_values, training_error_values, validataion_error_values, iters):
    # Visualiziation of Loss, Training Error, and Validation Error
    fig1, ax1 = plt.subplots()
    ax1.plot(iters, loss_values, 'r', label='Training Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(iters, training_error_values, 'g', label='Training Error')
    ax2.plot(iters, validataion_error_values, 'b', label='Validation Error')
    ax2.set_ylabel('Error/cm')

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper center')
    plt.show()

if __name__ == "__main__":
    tel, ty, vel, vy = normalizeData(train_eye_left, train_y, val_eye_left, val_y)
    lv, ter, ver, itr = trainNetwork(3000, tel, ty, vel, vy)
    visualizeTraining(lv, ter, ver, itr)
