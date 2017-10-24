import tensorflow as tf
import gym
import numpy as np

def generate_rollouts(env, pro_model, adv_model, num_paths, horizon):
	"""
	generate rollout data under env for num_paths * horizon.
	"""
	# just so we get the type of both pro actions and adv actions for future use.
	action = env.sample_action()

	pro_act_type = action.pro
	adv_act_type = action.adv

	# this list should contains num_paths dictionaries with rollout data from each path.
	paths = []

	for i in range(num_paths):
		path = {}
		states = []
		pro_actions = []
		adv_actions = []
		pro_rewards = []
		adv_rewards = []

		# initialize the env
		state = env.reset()
		states.append(state)

		for j in range(horizon):
			pro_act = pro_model.predict(state)
			adv_act = adv_model.predict(state)
			pro_actions.append(pro_act)
			adv_actions.append(adv_act)

			action.pro = pro_act
			action.adv = adv_act

			state, reward, done, _ = env.step(action)

			pro_rewards.append(reward)
			adv_rewards.append(-reward)

			if not done:
				states.append(state)
			else:
				state = env.reset()
				states.append(state)

		path['states'] = np.array(states)
		path['pro_actions'] = np.array(pro_actions)
		path['adv_actions'] = np.array(adv_actions)
		path['pro_rewards'] = np.array(pro_rewards)
		path['adv_rewards'] = np.array(adv_rewards)
		paths.append(path)

	return paths

def rarl_main(itrs, env, num_paths, horizon, learning_rate=5e-3, n_layers=2, size=500, activation=tf.tanh, output_activation=None):
	tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

	sess = tf.Session(config=tf_config)
	sess.__enter__() # equivalent to `with sess:`

	pro_model = TRPO(env, sess)
	adv_model = TRPO(env, sess, pro=False)

	for i in range(itrs):
		paths = generate_rollouts(env, pro_model, adv_model, num_paths, horizon)
		# TODO: 1. Concate rollouts into proper form to feed in neuralnet.
		#		2. compute advantages based on rewards.
		pro_model.learn(states, pro_actions, pro_advantages)
		adv_model.learn(states, adv_actions, adv_advantages)

		# TODO: compute and log avg return/reward for each rollout

		# TODO: save/load model/data when necessary; render when necessary; plot when necessary.

	return pro_model, adv_model


