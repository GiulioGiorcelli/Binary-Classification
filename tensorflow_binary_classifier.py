print("adagrad")
#Import untilities
print("-----------------------------")
print("Importing libraries")
import pandas as pd
import numpy as np
import teradata
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#Import Synthetic Minority Over-Sampling Technique algorithm
from imblearn.over_sampling import SMOTE

#Import Neural Network Libraries
import tensorflow as tf
import numpy as np
import time
from sklearn import metrics
print("Libraries imported")
print("-----------------------------\n")

#Import and massage data
print("-----------------------------")
print("Importing dataset")
df = pd.read_pickle('df_3')
print("Dataset Imported")
print("-----------------------------\n")



#Splitting train, test and validation
training_features, test_features, training_target, test_target, = train_test_split(df.drop(['APPLICATIONS'], axis=1),
                                               df['APPLICATIONS'],
                                               test_size = .015,
                                               random_state=12)

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = 0.02,
                                                  random_state=12)
												  
#Last clean-up before syntetic resampling
df['LOAN_AMT_NEEDED'] = df['LOAN_AMT_NEEDED'].fillna(df['LOAN_AMT_NEEDED'].mean())
df['DAY_PHONE_MSA_POPULATION'] = df['DAY_PHONE_MSA_POPULATION'].fillna(df['DAY_PHONE_MSA_POPULATION'].mean())
df['HOME_VALUE'] = df['HOME_VALUE'].fillna(df['HOME_VALUE'].mean())
df['PROPERTY_MORTGAGE_1_INT'] = df['PROPERTY_MORTGAGE_1_INT'].fillna(df['PROPERTY_MORTGAGE_1_INT'].mean())



from imblearn.over_sampling import SMOTE

#Using Synthetic Minority Over-Sampling techinque to balance the classes
print("-----------------------------")
print("Applying SMOTE to train and val subset")
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
x_val, y_val = sm.fit_sample(x_val, y_val)
print("Over-sampling completed")
print("-----------------------------\n")

#Check SMOTE results
print("Below are the results from the SMOTE algorithm")
print("Y=1 in raw train subset:", y_train.sum()) 
print("Y=1 in resampled train subset:", y_train_res.sum())
print("\n")

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = 10 # Number of feature
n_classes = 2 # Number of classes to predict

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
	
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()	


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # batch_y.shape = (batch_y.shape[0], 1)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
global result 
result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})