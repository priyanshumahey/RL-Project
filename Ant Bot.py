import gym
import numpy as np
import pybullet as p
import pybulletgym.envs
import time


def relu(x):
    return np.maximum(x, 0)


class SmallReactivePolicy:
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, observation_space, action_space):
        assert weights_dense1_w.shape == (observation_space.shape[0], 128)
        assert weights_dense2_w.shape == (128, 64)
        assert weights_final_w.shape  == (64, action_space.shape[0])

    @staticmethod
    def act(ob):
        x = ob
        x = relu(np.dot(x, weights_dense1_w) + weights_dense1_b)
        x = relu(np.dot(x, weights_dense2_w) + weights_dense2_b)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x


def main():
    print("create env")
    env = gym.make("AntPyBulletEnv-v0")
    env.render(mode="human")
    pi = SmallReactivePolicy(env.observation_space, env.action_space)

    env.reset()
    torsoId = -1
    for i in range(p.getNumBodies()):
        print(p.getBodyInfo(i))
        if p.getBodyInfo(i)[0].decode() == "torso":
           torsoId = i
           print("found torso")

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        print("frame")
        
        while 1:
            time.sleep(0.02)
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1      
            print("reward")
            print(r)
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            still_open = env.render("human")
            if still_open is None:
                return
            if not done:
                continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0:
                    break



weights_dense1_w = np.random.randn(28,128)
weights_dense1_b = np.random.randn(128,)
weights_dense2_w = np.random.randn(128,64)
weights_dense2_b = np.random.randn(64,)
weights_final_w= np.random.randn(64, 8)
weights_final_b= np.random.randn(8,)



if __name__ == "__main__":
    main()

