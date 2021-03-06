"""TTT4275 Project in Classification"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import style
import matplotlib.animation as animation
from sklearn.metrics import ConfusionMatrixDisplay

style.use('seaborn-paper')

legends = ["Setosa", "Versicolour", "Virginica"]

attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_data = [
    pd.read_csv("./Iris_TTT4275/class_1", names = attribute_names),
    pd.read_csv("./Iris_TTT4275/class_2", names = attribute_names),
    pd.read_csv("./Iris_TTT4275/class_3", names = attribute_names)]

n_attributes = len(attribute_names)
n_classes = len(iris_data)
dropped_attributes = [] #Any listed attribute dropped.
n_dropped = len(dropped_attributes)
n_attributes -= n_dropped
split_at = 30
alpha = 0.005
iterations = 500
test_first = True
tolerated_err = 0.045

def plot_histogram(iris_data):
    bins = np.linspace(0, 10, 100)

    for count, value in enumerate(iris_data):
        value["class"] = count

    fig, axis = plt.subplots(n_attributes)

    for i in range(n_attributes):
        for j in range(n_classes):
            legend = legends[j]
            axis[i].hist(iris_data[j][attribute_names[i]].to_numpy(), bins=bins,histtype="barstacked", alpha=0.5, label="Class " + legend, edgecolor="k")
        axis[i].legend(loc='upper right')
        axis[i].set_title(attribute_names[i])
        axis[i].set_ylabel("n")
        axis[i].set_xlabel("mm")
    axis[0].set_ylim([0, 15])
    axis[1].set_ylim([0, 15])
    plt.show()  

plot_histogram(iris_data)

def drop_attributes(iris_data, dropped_attributes):
    for i in range(len(iris_data)):
        iris_data[i]["class"] = i
        iris_data[i] = iris_data[i].drop(columns=dropped_attributes)

    return(iris_data)
    
iris_data = drop_attributes(iris_data, dropped_attributes)

def split_data(iris_data, split_at, default):
    if default:
        train_x = np.concatenate([df.iloc[0:split_at, :-1].to_numpy() for df in iris_data])

        train_y_labels = np.concatenate([df.iloc[0:split_at,-1].to_numpy() for df in iris_data])

        train_y = np.zeros((train_y_labels.shape[0], n_classes))
        for i, label in np.ndenumerate(train_y_labels):
            train_y[i][round(label)] = 1

        test_x = np.concatenate([df.iloc[split_at:, :-1].to_numpy() for df in iris_data])

        test_y_labels = np.concatenate([df.iloc[split_at:,-1].to_numpy() for df in iris_data])

        test_y = np.zeros((test_y_labels.shape[0],n_classes))
        for i, label in np.ndenumerate(test_y_labels):
            test_y[i][round(label)] = 1
    else:
        train_x = np.concatenate([df.iloc[split_at:,:-1].to_numpy() for df in iris_data])

        train_y_labels = np.concatenate([df.iloc[split_at:,-1].to_numpy() for df in iris_data])

        train_y = np.zeros((train_y_labels.shape[0], n_classes))
        for i, label in np.ndenumerate(train_y_labels):
            train_y[i][round(label)] = 1

        test_x = np.concatenate([df.iloc[0:split_at, :-1].to_numpy() for df in iris_data])

        test_y_labels = np.concatenate([df.iloc[0:split_at,-1].to_numpy() for df in iris_data])

        test_y = np.zeros((test_y_labels.shape[0],n_classes))
        for i, label in np.ndenumerate(test_y_labels):
            test_y[i][round(label)] = 1
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = split_data(iris_data, split_at, test_first)

W = np.zeros((n_classes, n_attributes))
w_0 = np.zeros((n_classes,))


def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))

def prediction(x, W, w_0):
    prediction = np.array([sigmoid((np.matmul(W,x[i])+w_0)) for i in range(x.shape[0])])
    return prediction


#Mean square error
def MSE(pred, y):
    # (1/N)*sum(error^2)
    
    return np.sum(np.matmul(np.transpose(pred-y),(pred-y))) / pred.shape[0]

def train_W_b(pred, y, x, W, w_0, alpha):
    pred_error = pred - y
    pred_z_error = np.multiply(pred,(1-pred))
    squarebracket = np.multiply(pred_error, pred_z_error)

    dW = np.zeros(W.shape)
    # Gradient of MSE with respect to W
    for i in range (x.shape[0]):
        dW = np.add(dW, np.outer(squarebracket[i], x[i]))
    
    dw_0 = np.sum(squarebracket, axis=0)

    return ((W-alpha*dW), (w_0-alpha*dw_0))

figure = plt.figure()
axis = figure.add_subplot(1,1,1)

# Arrays for storing plot values
plot_iteration = []
train_errors = []
test_errors = []

error_rate = 1
def get_error_rate(x, y, W, w_0):
    pred = prediction(x, W, w_0)

    mistakes = 0

    for i in range(pred.shape[0]):
        if np.argmax(y[i]) != np.argmax(pred[i]):
            mistakes += 1

    return mistakes/pred.shape[0]

def gen():
    global error_rate
    i = 0
    while (error_rate >= tolerated_err and i<iterations):
        i += 1
        yield i

def run(i):
    global W
    global w_0
    global error_rate

    error_rate = get_error_rate(train_x, train_y, W, w_0)
    print(error_rate)

    train_prediction = prediction(train_x, W, w_0)
    test_prediction = prediction(test_x, W, w_0)

    train_error = MSE(train_prediction, train_y)
    test_error = MSE(test_prediction, test_y)
    #print(test_error)
    #print(train_error)

    plot_iteration.append(float(i))
    train_errors.append(train_error)
    test_errors.append(test_error)

    axis.clear()
    axis.plot(plot_iteration, train_errors, "blue")
    axis.plot(plot_iteration, test_errors, "red")

    W, w_0 = train_W_b(train_prediction, train_y, train_x, W, w_0, alpha)

# Generate confusion matrix
def generate_confusion_matrix(x, y, W, w_0):
    pred = prediction(x, W, w_0)

    confusion_matrix = np.zeros((n_classes, n_classes))

    for i in range(pred.shape[0]):
        confusion_matrix[np.argmax(y[i])][np.argmax(pred[i])] += 1

    return confusion_matrix


animate = animation.FuncAnimation(figure, run, interval=16, frames=gen, repeat = False)
print(W)
plt.show()


print("Error rate train")
print(get_error_rate(train_x, train_y, W, w_0))
print("Confusion matrix train")
confusion_matrix_train = generate_confusion_matrix(train_x,train_y, W, w_0)
print(confusion_matrix_train)
disp = ConfusionMatrixDisplay(confusion_matrix_train, display_labels=legends)
disp.plot()

print("Error rate test")
print(get_error_rate(test_x, test_y, W, w_0))
confusion_matrix_test = generate_confusion_matrix(test_x,test_y, W, w_0)
print("Confusion matrix test")
print(confusion_matrix_test)
disp2 = ConfusionMatrixDisplay(confusion_matrix_test, display_labels=legends)
disp2.plot()
plt.show()