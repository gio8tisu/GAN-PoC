# Generative Adversarial Network Proof of Concept (GAN PoC)
Throgh this experiment I pretend to show how a naive aproach to mimic a
normal distribution fails, while on the other hand, an adversatial training succeeds.

This is a useless experiment ment to ilustrate the adversarial training set-up 
where I try to train a fully connected neural network (NN) to transform samples
drawn from a uniform distribution into "realistic" samples from a normal distribution.
The generator network is the same for the naive approach and the adversarial one.

## Traget distribution
The target distribution is a normal with mean 0 and variance 1.
The samples can be drawn once in the entire train process and then "re-used" or
can be drawn at each iteration.

## Generator network
The generator network samples from a uniform distribution with range -0.5,0.5.
It has a number of fully connected layers with ReLU activations.

## Naive approach
The naive approach takes the input (a sample from the uniform distribution),
passes it through the generator NN and compares this output with the target
(a sample from the normal distribution) using the mean square error (MSE) metric.
A succesfully trained model using this method with generate samples around the
target distribution's mean value. (?)

## Adversarial approach
The adversarial approach is similar to the naive one, but uses a discriminator NN
to measure the similarity with the target distribution.
I.e, another model is simultaniously trained to predict how probable its input
comes from the target distribution or it has been synthesized by the generator model
(hence, the name adversarial).
