# CS 534
# AI1 skeleton code
# By Quintin Pope
import pandas as pd


# Loads a data file from a provided file location.
def load_data(path):
    df = pd.read_csv (path)
    return df

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    #step 1: remove the id column
    preprocess_data = train.drop(columns='id')

    #step 2: remove the date column, but keep the data to use later
    dates = preprocess_data.pop('date')

    #step 3: add year, month and day columns from previously removed date column
    preprocess_data['year'] = [datetime.datetime.strptime(date, "%m/%d/%Y").year for date in dates]
    preprocess_data['month'] = [datetime.datetime.strptime(date, "%m/%d/%Y").month for date in dates]
    preprocess_data['day'] = [datetime.datetime.strptime(date, "%m/%d/%Y").day for date in dates]

    #step 4: add dummy column of just 1's
    preprocess_data['dummy'] = [1 for date in dates]

    

    return preprocessed_data

# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    # Your code here:

    return weights, losses

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    # Your code here:

    return

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:


# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



