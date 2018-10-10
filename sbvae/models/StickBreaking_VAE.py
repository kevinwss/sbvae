import numpy as np
import theano
import theano.tensor as T
import sys
#sys.path.append(r'/home/shaoswan/phd/beyasian/sbvae')

from models.variational_coders.encoders import StickBreakingEncoder
from models.variational_coders.decoders import Decoder

#from encoders import StickBreakingEncoder
#from decoders import Decoder

#--------------------------------------------
'''
class StickBreakingEncoder(object):
    def __init__(self, rng, input, batch_size, in_size, latent_size, W_a = None, W_b = None, epsilon = 0.01):
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.input = input
        
        # setup variational params
        if W_a is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, latent_size-1)), dtype=theano.config.floatX)
            W_a = theano.shared(value=W_values, name='W_a')
        if W_b is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, latent_size-1)), dtype=theano.config.floatX)
            W_b = theano.shared(value=W_values, name='W_b')
        self.W_a = W_a
        self.W_b = W_b

        # compute Kumaraswamy samples                                                                                                                                                      
        uniform_samples = T.cast(self.srng.uniform(size=(batch_size, latent_size-1), low=0.01, high=0.99), theano.config.floatX)
        self.a = Softplus(T.dot(self.input, self.W_a))
        self.b = Softplus(T.dot(self.input, self.W_b))
        v_samples = (1-(uniform_samples**(1/self.b)))**(1/self.a)

        # setup variables for recursion                                                                                                                                   
        stick_segment = theano.shared(value=np.zeros((batch_size,), dtype=theano.config.floatX), name='stick_segment')
        remaining_stick = theano.shared(value=np.ones((batch_size,), dtype=theano.config.floatX), name='remaining_stick')

        def compute_latent_vars(i, stick_segment, remaining_stick, v_samples):
            # compute stick segment                                                                                                     
            stick_segment = v_samples[:,i] * remaining_stick
            remaining_stick *= (1-v_samples[:,i])
            return (stick_segment, remaining_stick)

        (stick_segments, remaining_sticks), updates = theano.scan(fn=compute_latent_vars,
                                                                  outputs_info=[stick_segment, remaining_stick],sequences=T.arange(latent_size-1),
                                                                  non_sequences=[v_samples], strict=True)

        self.avg_used_dims = T.mean(T.sum(remaining_sticks > epsilon, axis=0))
        self.latent_vars = T.transpose(T.concatenate([stick_segments, T.shape_padaxis(remaining_sticks[-1, :],axis=1).T], axis=0))
        
        self.params = [self.W_a, self.W_b]
        

class Decoder(object):
    def __init__(self, rng, input, latent_size, out_size, activation, W_z = None, b = None):
        self.input = input
        self.activation = activation

        # setup the params                                                                                                                          
        if W_z is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(latent_size, out_size)), dtype=theano.config.floatX)
            W_z = theano.shared(value=W_values, name='W_hid_z')
        if b is None:
            b_values = np.zeros((out_size,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
        self.W_z = W_z
        self.b = b
        
        self.pre_act_out = T.dot(self.input, self.W_z) + self.b
        self.output = self.activation(self.pre_act_out)
        
        # gather parameters
        self.params = [self.W_z, self.b]

        '''
#--------------------------------------------
### Stick-Breaking VAE ###
class StickBreaking_VAE(object):
    def __init__(self, rng, input, batch_size, layer_sizes, 
                 layer_types, activations, latent_size, out_activation): # architecture specs
        
        # check lists are correct sizes
        assert len(layer_types) == len(layer_sizes) - 1
        assert len(activations) == len(layer_sizes) - 1
    
        # Set up the NN that parametrizes the encoder
        layer_specs = zip(layer_types, layer_sizes, layer_sizes[1:])
        self.encoding_layers = []
        next_layer_input = input
        activation_counter = 0        
        for layer_type, n_in, n_out in layer_specs:
            next_layer = layer_type(rng=rng, input=next_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out)
            next_layer_input = next_layer.output
            self.encoding_layers.append(next_layer)
            activation_counter += 1

        # init encoder
        self.encoder = StickBreakingEncoder(rng, input=next_layer_input, batch_size=batch_size, in_size=layer_sizes[-1], latent_size=latent_size)
        
        # init decoder 
        self.decoder = Decoder(rng, input=self.encoder.latent_vars, latent_size=latent_size, out_size=layer_sizes[-1], activation=activations[-1])

        # setup the NN that parametrizes the decoder (generative model)
        layer_specs = zip(reversed(layer_types), reversed(layer_sizes), reversed(layer_sizes[:-1]))
        self.decoding_layers = []
        # add output activation as first activation.  last act. taken care of by the decoder
        activations = [out_activation] + activations[:-1]
        activation_counter = len(activations)-1
        next_layer_input = self.decoder.output
        for layer_type, n_in, n_out in layer_specs:
            # supervised decoding layers
            next_layer = layer_type(rng=rng, input=next_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out)
            next_layer_input = next_layer.output
            self.decoding_layers.append(next_layer)
            activation_counter -= 1
            
        # Grab all the parameters--only need to get one half since params are tied
        self.params = [p for layer in self.encoding_layers for p in layer.params] + self.encoder.params + self.decoder.params + [p for layer in self.decoding_layers for p in layer.params]

        # Grab the posterior params
        self.post_a = self.encoder.a
        self.post_b = self.encoder.b

        # grab the kl-divergence functions
        self.calc_kl_divergence = self.encoder.calc_kl_divergence

        # Grab the reconstructions and predictions
        self.x_recon = next_layer_input
