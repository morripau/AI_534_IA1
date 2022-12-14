# CS 534
# AI1 skeleton code
# By Quintin Pope
import pandas as pd, os, datetime, copy as cp, matplotlib.pyplot as plt


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path, parse_dates= ["date"])
    return loaded_data

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize:bool=False, drop_sqrt_living15:bool=False):
    # Your code here:
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data.insert(0, "bias", 1)
    data.drop(["id", "date"], axis=1, inplace=True)
    
    
    age_since_renovated = np.zeros(len(data))
    
    for (i, value) in enumerate(data["yr_renovated"]):
        if value==0:
            age_since_renovated[i] = data["year"][i] - data["yr_built"][i]
        else:
            age_since_renovated[i] = data["year"][i] - data["yr_renovated"][i]
            
    data["age_since_renovated"] = age_since_renovated
    
    
    if normalize:
        global params
        params = {}
        for col in data.columns:
            if col != "price" and col!= "waterfront" and col != "bias":
                μ = np.mean(data[col])
                σ = np.std(data[col])
                data[col] = (data[col] -μ)/ σ
                params[col] = (μ, σ)
                
    if drop_sqrt_living15:
        data.drop(sqrt_living15, axis=1, inplace=True)
     
    preprocessed_data = data.copy()  

    return preprocess_data
    

###
### This piece was included in preprocess_data above
###
# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
#def modify_features(data):
    # Your code here:

   # return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    # Your code here:
    n_data = data.shape[0]
    n_features = data.shape[1]
    n_iter = 100
    x = np.transpose(data)
    w = np.random.rand(n_features)
    #w = np.zeros(n_features)
    losses = np.zeros(n_iter)
        
    for i in range(n_iter):
        grad = np.zeros(n_features)
        loss = 0
        for j in range(n_data):
            grad[:] += (np.dot(w,x[:,j])-labels[j])*x[:,j]
            loss += (np.dot(w,x[:,j])-labels[j])**2
        w -= lr*2/n_data*grad
        
        losses[i] = 1/n_data*loss
        
    weights = w

    return weights, losses

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, learning_rates):
    fig, ax = plt.subplots()
    plt.ylabel('loss')
    plt.xlabel('iteration')

    for i, loss in enumerate(losses):
        line = plt.plot(loss, label=f"γ={learning_rates[i]}")
        
    ax.legend(loc='upper right')
    fig

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.



# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:



# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



