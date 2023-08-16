import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

'''
采用神经网络识别苹果，训练集由分辨率均为196pixel×196pixel的10张苹果图片组成
程序测试表明：归一化、赋初值、学习率的设定是影响收敛与否的关键因素
正反样本的选择会提高预测准确率
'''

# 加载图片，生成训练集或测试集，并对数据进行预处理
def load_image_data(dir_data):
	names = os.listdir(dir_data)
	data_set_x_orig = np.zeros([len(names),196,196,3])  # 用于存储所有图片数据集

	# （1）获取数据集-4维数据
	index = 0
	for img_name in (names):
		if os.path.splitext(img_name)[1] == '.jpg':  # 仅选择.jpg格式的图片
			img = Image.open(dir_data + '\\' + img_name)
			data_set_x_orig[index] = np.array(img)
                        
			index = index + 1
			print('data_set shape = ', data_set_x_orig.shape, 'img size = ', img.size, 'img_dpi=', img.info['dpi'])
                        
			#plt.imshow(img_arr)
			#plt.show()
                        
	# （2）重塑数据集-2维数据
	data_set_x_orig_flatten = data_set_x_orig.reshape(data_set_x_orig.shape[0],-1).T

	# （3）标准化数据集-2维数据
	data_set_x = data_set_x_orig_flatten/255
	
	return data_set_x


# Graded function: sigmoid

def sigmoid(z):
    """
    This function compute the sigmoid of z

    Argument:
    z -- a scalar or numpy array

    Return：
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))
    
    return s


# Graded function: Initialization

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


# Graded function: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-apple, 1 if apple) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)            # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))         # compute cost
 
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)  # remove axes of length one from a numpy array
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


# Graded function: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        print('loop = ', i,'dw=',dw[0], 'db=',db, 'cost =', cost )

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# Graded function: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

    
# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations = 100, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
      
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    print('Y_prediction_train=',Y_prediction_train, 'Y_prediction_test=',Y_prediction_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    print('w=', w, '\n', 'b=',b)

    return d


if __name__ == '__main__':
    # Get the data set for training and testing
    X_train = load_image_data("apple_train")
    Y_train = np.array([0,1,0,1,0,1,1,1,1])
    X_test = load_image_data("apple_test")
    Y_test = np.array([1,1,0,0,0,1,1,1,1])

    # Get the model parameters
    print('X_train=',X_train.shape, '\n X_test=', X_test.shape)
    apple_model = model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.003, print_cost = False)
    costs_model = apple_model["costs"]
    num_iterations =  apple_model["num_iterations"]

    # plot the costs with respect to num_iteration
    plt.plot(np.arange(num_iterations/100),costs_model)
    plt.show()
