import gym
from gym import wrappers
import numpy as np
import random
import scipy


# Calculates the Standard Deviation
def find_SD(index, list):
    smallest_value = list[0][index]
    largest_value = list[0][index]
    for item in list:
        if item[index] < smallest_value:
            smallest_value = item[index]
        if item[index] > largest_value:
            largest_value = item[index]
    return largest_value - smallest_value


env = gym.make("CartPole-v0")
env = wrappers.Monitor(env, '/tmp/cartpole-cross-entropy-1',force=True)

# Mean of the deviation
cart_position_M = random.random() - 0.5
cart_velocity_M = random.random() - 0.5
pole_angle_M = random.random() - 0.5
pole_velocity_M = random.random() - 0.5

# Standard Deviation
cart_position_SD = 1
cart_velocity_SD = 1
pole_angle_SD = 1
pole_velocity_SD = 1

for batch in range(50):

    samples = []
    # The batch
    for iteration in range(100):

        # Choose the current sample
        cart_position_C = np.random.normal(cart_position_M, cart_position_SD)
        cart_velocity_C = np.random.normal(cart_velocity_M, cart_velocity_SD)
        pole_angle_C = np.random.normal(pole_angle_M, pole_angle_SD)
        pole_velocity_C = np.random.normal(pole_velocity_M, pole_velocity_SD)

        # resets the state
        observation = env.reset()
        cart_position = observation[0]
        cart_velocity = observation[1]
        pole_angle = observation[2]
        pole_velocity = observation[3]
        total_reward = 0
        done = False

        # The game
        while done == False:
            # env.render()
            # Choose action
            if cart_position * cart_position_C + \
                            cart_velocity * cart_velocity_C + \
                            pole_angle * pole_angle_C + \
                            pole_velocity * pole_velocity_C \
                    > 0:
                observation, reward, done, info = env.step(1)
            else:
                observation, reward, done, info = env.step(0)
            total_reward += reward
            cart_position = observation[0]
            cart_velocity = observation[1]
            pole_angle = observation[2]
            pole_velocity = observation[3]
        # saves the most recent sample
        samples += [[total_reward, cart_position_C, cart_velocity_C, pole_angle_C, pole_velocity_C]]

    '''iterates through the entire sample list, finds the 10 largest, and saves them.'''
    largest_samples = [samples[0]]
    for item in samples:
        for index, sample in enumerate(largest_samples):
            if item[0] > sample[0]:
                if len(largest_samples) > 10:
                    largest_samples[index] = item
                    item = sample
                else:
                    largest_samples += [item]
                    break

    # Mean of the deviation
    cart_position_M = 0
    cart_velocity_M = 0
    pole_angle_M = 0
    pole_velocity_M = 0

    smallest_values = [largest_samples[0][1]]
    largest_cp = 0
    largest_cv = 0
    largest_pa = 0
    largest_pv = 0
    # Calculates the new mean and standard deviation
    for index, sample in enumerate(largest_samples):
        cart_position_M += sample[1]
        cart_velocity_M += sample[2]
        pole_angle_M += sample[3]
        pole_velocity_M += sample[4]

    cart_position_M /= 10
    cart_position_M /= 10
    cart_velocity_M /= 10
    pole_angle_M /= 10
    pole_velocity_M /= 10

    # Standard Deviation
    cart_position_SD = find_SD(1, largest_samples)
    cart_velocity_SD = find_SD(2, largest_samples)
    pole_angle_SD = find_SD(3, largest_samples)
    pole_velocity_SD = find_SD(4, largest_samples)

env.close()
gym.upload('/tmp/cartpole-cross-entropy-1', api_key='sk_4cMtKbiGR8SWoX3rEQnVRw')
