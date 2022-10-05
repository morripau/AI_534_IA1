# CS 534
# AI1 skeleton code
# By Quintin Pope
import pandas as pd, os, numpy as np, matplotlib.pyplot as plt, copy as cp


# Loads a data file from a provided file location.
def load_data(path):
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
        data.drop(["sqrt_living15"], axis=1, inplace=True)
     
    preprocessed_data = data.copy()    
    return preprocessed_data

    

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
def plot_losses(losses):
    fig, ax = plt.subplots()
    plt.ylabel('loss')
    plt.xlabel('iteration')
    col = ['r', 'g', 'b', 'c', 'm']

    for i, (γ, loss) in enumerate(losses.items()):
        iterations = [j for j in range(0, len(loss))]
        plt.plot(iterations, loss, label=f"γ={γ}", color=col[i])
        
    ax.legend(loc='upper right')
    plt.show()

def compare_rate(data, labels, learning_rates):
    loss = {}
    for lr in learning_rates:
        loss[f"{lr}"] = gd_train(data, labels, lr)[1]
        
    return loss

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
training_file_dir = os.path.join(os.getcwd(), "IA1_train.csv")
train_data = load_data(training_file_dir)


valid_file_dir = os.path.join(os.getcwd(), "IA1_dev.csv")
valid_data = load_data(valid_file_dir)

data = {
    "train" :  preprocess_data(cp.deepcopy(train_data)),
    "train_norm" : preprocess_data(cp.deepcopy(train_data), normalize=True),
    "valid" : preprocess_data(cp.deepcopy(valid_data)),
    "valid_norm" : preprocess_data(cp.deepcopy(valid_data), normalize=True)
}


# Part 1 . Implement batch gradient descent and experiment with different learning rates.
converging_learning_rates = [10**(-i) for i in [0, 1, 2, 3, 4]]
diverging_learning_rates = [10**(-i) for i in [0, 1]]

# part 1a - using normalized data
y_norm = data["train_norm"]["price"]

"""
weights_norm = [0]*len(learning_rates)
losses_norm = [0]*len(learning_rates)

for i, γ in enumerate(learning_rates):
    weights_norm[i] = gd_train(data["train_norm"], y_norm, γ)[1]
    losses_norm[i] = gd_train(data["train_norm"], y_norm, γ)[2]

"""
losses_norm = compare_rate(data["train_norm"].to_numpy(), y_norm.to_numpy(), converging_learning_rates)

#print(losses_norm)
plot_losses(losses_norm)

# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



