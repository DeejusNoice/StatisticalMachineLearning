
# coding: utf-8

# In[2]:


import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='6860'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!\n')
    
    ### Training Data ###
    # Feature 1 extraction: Average Brightness
    avg_bness0 = average_brightness(train0)
    avg_bness1 = average_brightness(train1)
    
    
    # Feature 2 extraction: Standard Deviation
    var0 = std(train0)
    var1 = std(train1)
    
    
    # Calculate important parameters
    meanf1_0 = numpy.average(avg_bness0)
    varf1_0 = numpy.var(avg_bness0)
    meanf2_0 = numpy.average(var0)
    varf2_0 = numpy.var(var0)
    meanf1_1 = numpy.average(avg_bness1)
    varf1_1 = numpy.var(avg_bness1)
    meanf2_1 = numpy.average(var1)
    varf2_1 = numpy.var(var1)
    
    parameters = numpy.array([meanf1_0, varf1_0, meanf2_0, varf2_0,
                              meanf1_1, varf1_1, meanf2_1, varf2_1])
    
    
    # Calculate probabilites
    prior_y = 0.5
    
    prob_f1_0_y0 = calc_probabilities(avg_bness0, meanf1_0, varf1_0)
    prob_f1_0_y1 = calc_probabilities(avg_bness0, meanf1_1, varf1_1)
    prob_f2_0_y0 = calc_probabilities(var0, meanf2_0, varf2_0)
    prob_f2_0_y1 = calc_probabilities(var0, meanf2_1, varf2_1)
    
    prob_0_y0 = prior_y*prob_f1_0_y0*prob_f2_0_y0
    prob_0_y1 = prior_y*prob_f1_0_y1*prob_f2_0_y1
    
    prob_f1_1_y1 = calc_probabilities(avg_bness1, meanf1_1, varf1_1)
    prob_f1_1_y0 = calc_probabilities(avg_bness1, meanf1_0, varf1_0)
    prob_f2_1_y1 = calc_probabilities(var1, meanf2_1, varf2_1)
    prob_f2_1_y0 = calc_probabilities(var1, meanf2_0, varf2_0)
    
    prob_1_y1 = prior_y*prob_f1_1_y1*prob_f2_1_y1
    prob_1_y0 = prior_y*prob_f1_1_y0*prob_f2_1_y0
    
    # Predict class
    pred_y1 = predict_y(prob_1_y0, prob_1_y1)
    y1 = numpy.ones((len(train1), 1))
    accuracy(pred_y1, y1)
    
    pred_y0 = predict_y(prob_0_y0, prob_0_y1)
    y0 = numpy.zeros((len(train0), 1))
    accuracy(pred_y0, y0)
    
    
    
    ### Test data ###
    # Feature 1 extraction: Average Brightness
    test_bness0 = average_brightness(test0)
    test_bness1 = average_brightness(test1)
    
    # Feature 2 extraction: Standard Deviation
    test_var0 = std(test0)
    test_var1 = std(test1)
    
    # Calculate probabilities
    t_prob_f1_0_y0 = calc_probabilities(test_bness0, meanf1_0, varf1_0)
    t_prob_f1_0_y1 = calc_probabilities(test_bness0, meanf1_1, varf1_1)
    t_prob_f2_0_y0 = calc_probabilities(test_var0, meanf2_0, varf2_0)
    t_prob_f2_0_y1 = calc_probabilities(test_var0, meanf2_1, varf2_1)
    
    t_prob_0_y0 = prior_y*t_prob_f1_0_y0*t_prob_f2_0_y0
    t_prob_0_y1 = prior_y*t_prob_f1_0_y1*t_prob_f2_0_y1
    
    t_prob_f1_1_y1 = calc_probabilities(test_bness1, meanf1_1, varf1_1)
    t_prob_f1_1_y0 = calc_probabilities(test_bness1, meanf1_0, varf1_0)
    t_prob_f2_1_y1 = calc_probabilities(test_var1, meanf2_1, varf2_1)
    t_prob_f2_1_y0 = calc_probabilities(test_var1, meanf2_0, varf2_0)
    
    t_prob_1_y1 = prior_y*t_prob_f1_1_y1*t_prob_f2_1_y1
    t_prob_1_y0 = prior_y*t_prob_f1_1_y0*t_prob_f2_1_y0
    
    # Predict class
    t_pred_y1 = predict_y(t_prob_1_y0, t_prob_1_y1)
    t_y1 = numpy.ones((len(test1), 1))
   
    
    t_pred_y0 = predict_y(t_prob_0_y0, t_prob_0_y1)
    t_y0 = numpy.zeros((len(test0), 1))
    
    for x in parameters:
        print(x)
    print(accuracy(t_pred_y0, t_y0))
    print(accuracy(t_pred_y1, t_y1))


# Calculates the average brightness of the pixels of each image
# of an array of images
def average_brightness(images):
    avg_bness = numpy.zeros((len(images), 1))
    for x in range(len(images)):
        avg_bness[x] = numpy.average(images[x])
    return avg_bness
    

# Calculates the variance of the brightness of pixels
# of each image in an array of images
def variance(images):
    var = numpy.zeros((len(images), 1))
    for x in range(len(images)):
        var[x] = numpy.var(images[x])
    return var

def std(images):
    std = numpy.zeros((len(images), 1))
    for x in range(len(images)):
        std[x] = numpy.std(images[x])
    return std


# Calculates P(x|y) for a Gaussian probability distribution
def calc_probabilities(feature_matrix, mean, var):
    norm = 1/(numpy.sqrt(var*2*numpy.pi))
    prob = norm*numpy.exp(-((mean-feature_matrix)**2)/(2*var))
    return prob


# Predicts if X belongs to y=0 or y=1
def predict_y(prob_y0, prob_y1):
    pred_y = numpy.zeros((len(prob_y0), 1))
    for x in range(len(prob_y0)):
        if prob_y1[x] > prob_y0[x]:
            pred_y[x] = 1
    return pred_y


# Calculate accuracy of predicted values
def accuracy(pred, actual):
    correct = numpy.equal(pred, actual)
    num_correct = numpy.count_nonzero(correct)
    return num_correct/len(actual)



if __name__ == '__main__':
    main()

