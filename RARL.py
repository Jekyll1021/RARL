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

# def main(itrs, num_paths, )

