# CS 534
# AI1 skeleton code
# By Quintin Pope
from pickle import TRUE
import pandas as pd, os, numpy as np, matplotlib.pyplot as plt


# Loads a data file from a provided file location.
def load_data(path):
    loaded_data = pd.read_csv(path, parse_dates= ["date"])
    return loaded_data

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize:bool=False, drop_sqft_living15:bool=False, test:bool=False):
    preprocessed_data = data.copy()  

    # add year, month and day columns in the data DF
    preprocessed_data["year"]  = preprocessed_data["date"].dt.year
    preprocessed_data["month"] = preprocessed_data["date"].dt.month
    preprocessed_data["day"]   = preprocessed_data["date"].dt.day

    # add a bias column of all 1's, drop the date column
    preprocessed_data.insert(0, "bias", 1)
    preprocessed_data.drop(["id", "date"], axis=1, inplace=True)

    # add age since renovated column accounting for houses not renovated since built
    age_since_renovated = np.zeros(len(preprocessed_data))
    for (i, value) in enumerate(preprocessed_data["yr_renovated"]):
        if value==0:
            age_since_renovated[i] = preprocessed_data["year"][i] - preprocessed_data["yr_built"][i]
        else:
            age_since_renovated[i] = preprocessed_data["year"][i] - preprocessed_data["yr_renovated"][i]
    preprocessed_data["age_since_renovated"] = age_since_renovated

    # normalize data if option to normalize selected
    if normalize:
        global params
        params = {}
        for col in preprocessed_data.columns:
            if col != "price" and col!= "waterfront" and col != "bias":
                μ = np.mean(preprocessed_data[col])
                σ = np.std(preprocessed_data[col])
                preprocessed_data[col]   = (preprocessed_data[col] -μ)/ σ
                params[col] = (μ, σ)
    if test:
         for col in preprocessed_data.columns:
            if col != "price" and col != "waterfront" and col != "bias":
                μ = params[col][0]
                σ = params[col][1]
                preprocessed_data[col]   = (preprocessed_data[col] -μ)/ σ
                
    
    # drop sqrt living 15 if option selected to drop
    if drop_sqft_living15:
        preprocessed_data.drop(["sqft_living15"], axis=1, inplace=True)
    
    # return preprocessed data   
    return preprocessed_data



# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, lr, ϵ, n_iter=4000):
    # initialize variables
    labels      = data["price"].to_numpy()
    data        = data.drop(["price"], axis=1)
    data        = data.to_numpy()
    n_data      = data.shape[0]
    n_features  = data.shape[1]
    x           = np.transpose(data)
    weights     = np.ones(n_features)
    mse         = np.zeros(n_iter + 1)   
    convergence = {}

    # main iteration loop
    for i in range(n_iter):
        Bₙ   = np.zeros(n_features)
        loss = 0
        # loop through each data label
        for j in range(n_data):
            yᵢ    = labels[j]
            xᵢ    = x[:,j]
            ŷᵢ    = np.dot(weights, xᵢ)
            Bₙ[:] += (ŷᵢ - yᵢ) * xᵢ          
            loss += (ŷᵢ - yᵢ) ** 2
        

        ΔL       = (2/n_data) * Bₙ
        mse[i+1] = loss/n_data
        diff     = mse[i+1] - mse[i]
        
        # after each iteration check for convergence
        if abs(diff)<=ϵ and abs(diff) != mse[i+1]:
            weights -= lr * ΔL
            convergence[f"{lr}"] = True 
            return weights, mse[1: i+2], convergence
        
        elif diff < 0:
            weights -= lr * ΔL
            convergence[f"{lr}"] = "Will Converge"
            continue
        
        elif diff>0 and abs(diff) != mse[i+1]:
            convergence[f"{lr}"] = False
            return weights, mse[1: i+2], convergence
            
        weights -= lr * ΔL 
        
    return weights, mse[1: ], convergence

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, convergence, normalized=True):
    ax = plt.subplots(figsize=[16,9])
    plt.ylabel('Loss', fontweight="bold", )
    plt.xlabel('Iteration', fontweight="bold")
    #plt.xlim(1, 4000)
    if normalized:
        title_string = "normalized"
    else:
        title_string = "not normalized"

    plt.title(f"Convergence of Learning steps, {title_string}", fontweight="bold")

    for (γ, loss) in losses.items():
        if convergence[γ][γ]==False:
            pass
        else:
            iterations = [j for j in range(1, len(loss)+1)]
            plt.plot(iterations, loss, label=f"γ={γ}")
            
    ax.legend(loc='upper right')
    plt.savefig("convergence_plot")
    plt.show()

# runs multiple experiments given a set of learning rates (lrs)
# outputs a dictionary for weights, losses and convergences where
# the key is the learning rate.
def compare_rate(data, lrs, ϵ):
    # initialize weight MSE and convergence dictionaries
    mses         = {}
    convergences = {}
    weights      = {}
    # loop through learning rates and collect weights, MSE values and convergence
    for lr in lrs:
        (weight, mse, convergence) = gd_train(data, lr, ϵ)
        mses[f"{lr}"]              = mse
        convergences[f"{lr}"]      = convergence
        weights[f"{lr}"]           = weight
        
    return weights, mses, convergences

def validate(val_data, weights, convergences):
    val_labels = val_data["price"].to_numpy()
    val_data   = val_data.drop(["price"], axis=1)
    val_data   = val_data.to_numpy()
    n_data     = val_data.shape[0]
    x_val      = np.transpose(val_data)
    mse        = {}

    if type(weights) is dict:
        for (γ, weight) in weights.items():
            loss = 0
            if convergences[γ][γ]==False:
                pass
            else:
                for j in range(n_data):
                    yᵢ_val = val_labels[j]
                    xᵢ_val = x_val[:,j]
                    ŷᵢ_val = np.dot(weight, xᵢ_val)         
                    loss += (ŷᵢ_val - yᵢ_val) ** 2
                mse[f"{γ}"] = round((1/n_data) * loss, 3) 
    else:
        loss=0
        for j in range(n_data):
                yᵢ_val = val_labels[j]
                xᵢ_val = x_val[:,j]
                ŷᵢ_val = np.dot(weights, xᵢ_val)         
                loss  += (ŷᵢ_val - yᵢ_val) ** 2
        mse = round((1/n_data) * loss, 3)
        return mse
    
    return mse

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# step 1, load data
training_file_dir = os.path.join(os.getcwd(), "IA1_train.csv")
train_data        = load_data(training_file_dir)
valid_file_dir    = os.path.join(os.getcwd(), "IA1_dev.csv")
valid_data        = load_data(valid_file_dir)

# step 2, preprocess data
data = {
    "train" :  preprocess_data(train_data),
    "train_norm" : preprocess_data(train_data, normalize=True),
    "valid" : preprocess_data(train_data, test=True),
    "valid_norm" : preprocess_data(train_data, normalize=True, test=True),
    "train_redundancy_dropped" :  preprocess_data(train_data, drop_sqft_living15=True),
    "train_norm_redundancy_dropped" : preprocess_data(train_data, normalize=True, drop_sqft_living15=True),
    "valid_redundancy_dropped" : preprocess_data(train_data, drop_sqft_living15=True, test=True),
    "valid_norm_redundancy_dropped" : preprocess_data(train_data, normalize=True, drop_sqft_living15=True, test=True)
}

##########################################################################################
# Part 1 . Implement batch gradient descent and experiment with different learning rates.#
##########################################################################################
learning_rates_norm = [10**(-i) for i in [0, 1, 2, 3, 4]]

# this will be used as convergence criteria for part 2 as well
ϵ = 10**-3

# part 1a - using normalized data
weights_norm, mses_norm, convergences_norm = compare_rate(data["train_norm"], learning_rates_norm, ϵ)

# print(losses normalized)
plot_losses(mses_norm, convergences_norm)
print("Based on the normalized data plot, convergence occurs in all the plots except γ=1")

# validate normalized data
validation_norm = validate(data["valid_norm"], weights_norm, convergences_norm)
print("note validation data is presented as a dictionary in which learning rates are keys and MSE is value")
print(f"validation MSE from the learned weights for normalized data are: {validation_norm}")



#################################################################
# Part 2 a. Training and experimenting with non-normalized data.#
#################################################################

# after some experimenting, these set of learning rates seem to be right in the sweet spot
learning_rates_not_normalized = [10**(-i) for i in [0.001*j for j in range(10010, 10015)]]

# calculate weights, MSEs and convergences
weights, mses, convergences = compare_rate(data["train"], learning_rates_not_normalized, ϵ)

# print(losses not normalized)
plot_losses(mses, convergences, normalized=False)
print("Based on the non_normalized data plot, there appears to be a divergence threshold right around 9.85e-11")

# validate not normalized
validation = validate(data["valid"], weights, convergences)
print(f"validation MSE from the learned weights for NON normalized data are: {validation}")


#####################################################
# Part 2 b. Training with redundant feature removed.#
#####################################################
# note for the redundant feature drop we only looked at one learning rate, 0.1, to compare
data_sqft_living_dropped_result = gd_train(data["train_norm_redundancy_dropped"], 0.1, 10**-3)

data_sqft_living_dropped_weights = data_sqft_living_dropped_result[0]
data_sqft_living_dropped_convergences = data_sqft_living_dropped_result[2]

# show the weights for the dropped redundant feature at the ideal learning rate
print(f"The weights for learning rate 0.1 with the redundant feature are: {data_sqft_living_dropped_weights}")


# validate with the dropped redundante feature to compare
data_sqft_living_dropped_val = validate(data["valid_norm_redundancy_dropped"], data_sqft_living_dropped_weights, data_sqft_living_dropped_convergences)
print(f"validation MSE for the normalized data having dropped the redundance at learning rate 0.1 is: {data_sqft_living_dropped_val}")


