import numpy as np


#data i/o


data = open('read_it.txt','r').read()
name = data.lower()

name_set = set()
for chars in name:
    name_set.add(chars)  #take unique characters only

chars = list(name_set)
vocabulary_size = len(chars)
char_to_index = {ch: i for i, ch in enumerate(chars) }
index_to_chars = {i: ch for i, ch in enumerate(chars)}


f = open('read_it.txt','r')
lines = f.readlines()
count = 0  #number of names

for line in lines:
    count += 1

f.close()

#hyperparameters- fixed constants
num_of_neuron = 100
leaning_rate = .01

#parameters
Wxh = np.random.randn(num_of_neuron, vocabulary_size)*leaning_rate   #input to hidden layer
Whh = np.random.randn(num_of_neuron, num_of_neuron)*leaning_rate     #hidden layer to hidden layer
Why = np.random.randn(vocabulary_size, num_of_neuron)*leaning_rate   #hidden layer to output
bh = np.zeros((num_of_neuron, 1), dtype=float)                       #hidden bias
by = np.zeros((vocabulary_size, 1), dtype=float)                     #output bias

#memory variables for update_parameters
m_Wxh = np.zeros_like(Wxh)
m_Whh = np.zeros_like(Whh)
m_Why = np.zeros_like(Why)
m_bh = np.zeros_like(bh)     #hidden bias
m_by = np.zeros_like(by)     #output bias




def clip(gradients,min_value,max_value):

    dWhh, dWxh, dWhy, dbh, dby = gradients['dWhh'], gradients['dWxh'], gradients['dWhy'], gradients['dbh'], gradients['dby']

    for gradient in [dWhh, dWxh, dWhy, dbh, dby]:

        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                if gradient[i][j] > max_value:

                    gradient[i][j] = max_value
                elif gradient[i][j] < min_value:
                    gradient[i][j] = min_value


    gradients = {'dWhh':dWhh, 'dWxh':dWxh, 'dWhy':dWhy, 'dbh':dbh, 'dby':dby}

    return gradients



def TANH(z):
    return (np.exp(z)-np.exp(-1*z)) / (np.exp(z) + np.exp(-1*z))



def my_clip(x, min_value, max_value):

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > max_value:
                x[i][j] = max_value

            elif x[i][j] < min_value:
                x[i][j] = min_value

    return x

def softmax(x):

    e_x=np.exp(x - np.max(x))

    return e_x / np.sum(e_x, axis=0)



def cross_entropy_loss(y, y_s):

    #print(len(y_s))
    #print(type(y_s))
    #print(y_s)
    #print(sum(y_s))
   # print(len(y))
    #print(type(y))

    y_s = my_clip(y_s, 1e-10, 1. - 1e-10)
    n = y_s.shape[0]
    c_e = -np.sum(y * np.log(y_s + 1e-10)) / n

    return c_e




def rnn_forward(x, y, h_previous, parameters):
    #retrieve parameters
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    Why = parameters['Why']
    by = parameters['by']
    bh = parameters['bh']


    x_s, h_s, y_s= {}, {}, {}
    #print(h_previous.shape)
    h_s[-1] = np.copy(h_previous)
    x_s[0] = np.zeros((vocabulary_size, 1), dtype=float)
    #print(h_s)

    loss = 0
    #print(h_s)

    for t in range(len(x)):

        if t > 0:
            x_s[t] = np.zeros((vocabulary_size, 1), dtype=float)
            x_s[t][x[t]] = 1

        h_s[t] = TANH(Wxh.dot(x_s[t]) + Whh.dot(h_s[t - 1]) + bh)
        y_s[t] = softmax(Why.dot(h_s[t]) + by)


        #loss += -(-y[t]*np.log(y_s[t]) + (1-y[t])*np.log(1-y_s[t]))
        #loss += -np.log(y_s[t][y[t], 0])
        loss += cross_entropy_loss(y[t], y_s[t])

    #loss += cross_entropy_loss(y, y_s)
    cache = (x_s, h_s, y_s)

    return loss, cache

def rnn_backward(y, parameters, cache):

    #retrieve x_s,h_s,y_s
    x_s, h_s, y_s = cache

    #initialize all gradients to zero
    dh_next = np.zeros_like(h_s[0])

    gradients = {}

    for parameters_names in parameters.keys():
        gradients['d'+parameters_names] = np.zeros_like(parameters[parameters_names])


    for t in reversed(range(len(x_s))):
        dy = np.copy(y_s[t])
        dy[y[t]] -= 1
        gradients['dWhy'] = dy.dot(h_s[t].T)
        gradients['dby'] += dy

        dh=parameters['Why'].T.dot(dy) + dh_next
        d_tanh=(1 - h_s[t] ** 2) * dh

        gradients['dWhh'] += d_tanh.dot(h_s[t-1].T)
        gradients['dWxh'] += d_tanh.dot(x_s[t].T)
        gradients['dbh'] += d_tanh
        dh_next=parameters['Whh'].T.dot(d_tanh)

    h_previous=h_s[len(x_s)-1]

    return gradients,h_previous




def update_parameters(parameters,gradients):

    #uses rmsprop formula
    decay_rate=.99
    epsilon=1e-8

    #retieve parameters from 'parameters'

    Why=parameters['Why']
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    by=parameters['by']
    bh=parameters['bh']

    #retrieve parameters from 'gradients'
    dWhy = gradients['dWhy']
    dWhh = gradients['dWhh']
    dWxh = gradients['dWxh']
    dby = gradients['dby']
    dbh = gradients['dbh']

    #now update

    m_Why1 = np.multiply(m_Why, decay_rate) + (1-decay_rate) * dWhy ** 2
    Why -= leaning_rate * dWhy / (np.sqrt(m_Why1)  + epsilon)

    m_Whh1 = np.multiply(m_Whh, decay_rate) + (1 - decay_rate) * dWhh ** 2
    Whh -= leaning_rate * dWhh / (np.sqrt(m_Whh1) + epsilon)

    m_Wxh1 = np.multiply(m_Wxh, decay_rate) + (1 - decay_rate) * dWxh ** 2
    Wxh -= leaning_rate * dWxh / (np.sqrt(m_Wxh1) + epsilon)

    m_by1 = np.multiply(m_by, decay_rate) + (1 - decay_rate) * dby ** 2
    by -= leaning_rate * dby / (np.sqrt(m_by1) + epsilon)

    m_bh1 = np.multiply(m_bh, decay_rate) + (1 - decay_rate) * dbh ** 2
    bh -= leaning_rate * dbh / (np.sqrt(m_bh1) + epsilon)

    return parameters


def sample(parameters,char_to_index,seed):

    Whh, Wxh, Why, bh, by = parameters['Whh'], parameters['Wxh'], parameters['Why'], parameters['bh'], parameters['by']
    n_h=Whh.shape[1]  #number of column

    x = np.zeros((vocabulary_size, 1), dtype=float) #one hot vector x for 1st character
    h_previous = np.zeros((n_h,1))

    indices = []      #empty list that contains the list of indices of chars to generate
    index = -1        #flag to detect new line character

    counter = 0        #we will stop if we get 50 chars . it is used to break loop incase new line is not found
    newline=char_to_index['\n']

    while(index != newline and counter != 50):

        h = TANH(Whh.dot(h_previous) + Wxh.dot(x) + bh)
        z = Why.dot(h) + by
        y = softmax(z)

        np.random.seed(counter+seed)
        index = np.random.choice(list(range(vocabulary_size)), p = y.ravel() )
        indices.append(index)

        x = np.zeros(y.shape)
        x[index] = 1
        h_previous = h

        seed += 1
        counter += 1


    if(counter == 50):
        indices.append(newline)


    return indices

#single step forward propagation


def print_sample(index):
    text=''.join(index_to_chars[ix] for ix in index)
    print(text)


def optimise(x, y, h_previous, parameters):
    #print(h_previous.shape)

    loss, cache = rnn_forward(x, y, h_previous, parameters)

    gradients, h = rnn_backward(y,parameters, cache)

    gradients=clip(gradients, -5, 5)

    parameters = update_parameters(parameters, gradients)

    return loss, gradients, h[len(h)-1], parameters

def model(number_of_iteration):

    n_x = vocabulary_size
    n_h = num_of_neuron

    parameters = {'Wxh': Wxh, 'Whh': Whh, 'Why': Why, 'bh': bh, 'by': by}
    loss = -np.log(1.0/n_x)*n_h   #loss at iteration 0


    with open('read_it.txt') as f:
        samples=f.readlines()

    samples=[char.lower().strip() for char in samples]

    #shuffle list of all names

    np.random.seed(0)
    np.random.shuffle(samples)

    #initialize hidden state

    h_previous = np.zeros((n_h, 1), dtype=float)
    #print(h_previous.shape)
    #optimization loop

    for iter in range(number_of_iteration):
        index = iter % len(samples)
        x = [None] +[ char_to_index[ch] for ch in samples[index]]
        y = x[1:] + [char_to_index['\n']]

        #print(h_previous.shape)
        current_loss,gradients,h_prev,parameters=optimise(x, y, h_previous, parameters)

        loss = loss * .999 + current_loss * .001   #  smooth loss

        if iter % 5000 == 0:
            #print('iteration : %d , loss : %f ' % (iter, loss)+'\n')
            print(iter)
            print(loss)

            seed = 0
            for name in range(count):

                sampled_indices = sample(parameters,char_to_index,seed)
                print_sample(sampled_indices)
                seed += 1

            print('\n')

    return parameters

if __name__ == '__main__':
    parameters = model(90000)