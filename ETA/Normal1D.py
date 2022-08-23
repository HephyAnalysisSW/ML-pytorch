''' Normal distribution model 1D for equivariant tree algorithms
'''

import numpy as np
import scipy.stats

from groups import U1


def getEvents( N_events, N_channels, expected_bkg_yield=0., expected_signal_yield=100, sigma=.2, randomized_locations = True):
#if True:
#    N_events   = 100
#    N_channels = 10
#    expected_signal_yield = 100
#    sigma = .2
#    randomized_locations = True

    # Gaussian yields, centralized
    thr    =  np.arange(-1,1+2./N_channels,2./N_channels,dtype=float)

    # shift the mean by a random location  
    if randomized_locations:
        rnd_shift = 2*(-.5 + np.random.random(N_events))/N_channels
    else:
        rnd_shift = np.zeros(N_events)

    if expected_signal_yield>0:        
        yields = expected_signal_yield*np.array( [ [ scipy.stats.norm.cdf( (thr[i+1] + rnd_shift[i_event])/float(sigma)) - scipy.stats.norm.cdf( (thr[i] + rnd_shift[i_event])/float(sigma) ) for i in range( len(thr)-1 ) ] for i_event in range( N_events )] )
        #yields/=(np.sum(yields,axis=1).reshape(N_events,1))
        #yields*=norm
    else:
        yields = np.zeros( (N_events, N_channels) )

    # add flat background
    yields += expected_bkg_yield*np.ones( (N_events, N_channels) )

    result =  np.random.poisson( yields )

    # shift with a randomly chosen group element
    if randomized_locations:
        G      = U1(N_channels)
        result = np.matmul( G.elements[np.random.randint(G.N_elements,size=N_events)], result.reshape(N_events,N_channels,1)).reshape(N_events,N_channels)

    return result


#from scipy.stats import mvnun
#import numpy as np
#low = np.array([-10, -10])
#upp = np.array([.1, -.2])
#mu = np.array([-.3, .17])
#S = np.array([[1.2,.35],[.35,2.1]])
#p,i = mvnun(low,upp,mu,S)
#print (p)


