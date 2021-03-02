import pylab as pl
import scipy as sp
import numpy as np
from scipy.io import loadmat
import pdb


def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    imgs = data['data_patterns']
    labels = data['data_labels']
    return imgs, labels


	
def perceptron_train(X,Y,Xtest,Ytest,iterations=100,eta=.1):
    # initialize accuracy vector
    acc = sp.zeros(iterations)
    # initialize weight vector
    #X = np.insert(X, 0, 1, axis = 0) add bias
    #Xtest = np.insert(Xtest, 0, 1, axis = 0) add bias
    weights = np.random.uniform(-0.05,0.05,(256,))
    beta = float(np.random.uniform(-0.05,0.05,(1,)))
    #weights = np.insert(weights, 0, -beta, axis = 0) add bias

    # loop over iterations    
    for it in sp.arange(iterations):
        # find all indices of misclassified data
        predicted = np.dot(weights, X)
        predicted = np.where(predicted > 0, 1, -1)
        wrong = np.where(predicted == Y, 1, 0)
        result = np.where(wrong == 0)
        result = result[0]    
            # check if there really are misclassified data
        if result.shape[0] > 0:
            # pick a random misclassified data point
            rit = np.random.choice(result.shape[0], 1, replace=False)  
            rit = int(rit)
            # update weight vector
            weights += eta * Y[rit] * X[:, rit] #/ (it+1) prof said division by iter is not important # or 'weights +='
            # compute accuracy vector
            predicted_val = np.dot(weights, Xtest)
            predicted_val = np.where(predicted_val > 0, 1, -1)
            wrong_val = np.where(predicted_val == Ytest, 1, 0)
            acc[it] = np.sum(wrong_val) / Ytest.shape[0]
    # return weight vector and accuracy
    #return weights[1:],acc if have bias
    return weights,acc



def digits(digit):
    fname = "usps.mat"
    imgs,labels = load_data(fname)
    # we only want to classify one digit 
    labels = sp.sign((labels[digit,:]>0)-.5)

    # please think about what the next lines do
    permidx = sp.random.permutation(sp.arange(imgs.shape[-1]))
    trainpercent = 70.
    stopat = sp.floor(labels.shape[-1]*trainpercent/100.)
    stopat= int(stopat)

    # cut segment data into train and test set into two non-overlapping sets:
    X = imgs[:, 0:(2007 - 602)]
    Y = labels[0:(2007 - 602)]
    Xtest = imgs[:, (2007 - 602):]
    Ytest = labels[(2007 - 602):]
    #check that shapes of X and Y make sense..
    # it might makes sense to print them
    
    w,acc_perceptron = perceptron_train(X,Y,Xtest,Ytest)

    fig = pl.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(acc_perceptron*100.)
    pl.xlabel('Iterations')
    pl.title('Linear Perceptron')
    pl.ylabel('Accuracy [%]')

    # and imshow the weight vector
    ax2 = fig.add_subplot(1,2,2)
    # reshape weight vector
    weights = sp.reshape(w,(int(sp.sqrt(imgs.shape[0])),int(sp.sqrt(imgs.shape[0]))))
    # plot the weight image
    imgh = ax2.imshow(weights)
    # with colorbar
    pl.colorbar(imgh)
    ax2.set_title('Weight vector')
    # remove axis ticks
    pl.xticks(())
    pl.yticks(())
    # remove axis ticks
    pl.xticks(())
    pl.yticks(())

    # write the picture to pdf
    fname = 'Perceptron_digits-%d.pdf'%digit
    pl.savefig(fname)


digits(0)




