import torch
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from jsplendor.env import JsplendorEnv, FeatureExtractor
from jsplendor.utils import get_verbose_dict

def main():
    np.random.seed(1)
    verbose_dict = get_verbose_dict(True)

    load_model = False

    env = JsplendorEnv(verbose_dict)
    exp = 'transformer_model-9967'
    model_path = 'logs/{}/best_model'.format(exp)

    if load_model:
        model = PPO.load(model_path, env=env)
    else:
        policy_kwargs = dict(
                features_extractor_class=FeatureExtractor,
                net_arch=[64],
                activation_fn=torch.nn.ReLU)

        model = PPO("MlpPolicy", 
                    env, 
                    verbose=False,
                    policy_kwargs=policy_kwargs,
                    )
        trained_model = torch.load('./ckpt/model-9997.ckpt')
        
        state_dict = trained_model['state_dict']
        new_state_dict = dict()

        for key, value in state_dict.items():
            if 'model' in key:
                new_key = key[6::]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.policy.load_state_dict(new_state_dict)

    print('load model.')

    game_n = 100
    max_step = 100
    results = []
    for exp_i in range(game_n):
        obs, _ = env.reset(seed=exp_i)
        for i in range(max_step):
#            print('[step {}]'.format(i))
            # debug
            y1 = model.policy.extract_features(torch.from_numpy(obs).unsqueeze(dim=0).cuda().float())
            y2 = model.policy.mlp_extractor.policy_net(y1)
            y3 = model.policy.action_net(y2)
            torch.set_printoptions(sci_mode=False, precision=2)
            y4 = torch.softmax(y3, dim=1).detach().cpu().numpy()

            possible_actions = env.get_possible_actions()

            action_cand = y4 * possible_actions
            action = np.argmax(action_cand)
            
#            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
            if done:
                results.append(i)
                break

            if i==99:
                results.append(i)

    print(results)
    print('mean step: {}'.format(sum(results)/len(results)))
    
    fail = 0
    for result in results:
        if result == 100:
            fail += 1

    print('fail number: {}'.format(fail))

    print('done')


if __name__ == "__main__":
    main()
