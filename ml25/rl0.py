import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human")

div = 8
q = [[[0, 0, 0] for _ in range(div)] for _ in range(div)]

obs, info = env.reset(seed=42)
for i in range(1000000):
    x, v = obs
    if i % 500 == 0 or x > 0.4 or i > 100000:
        print(i)
        env.render()
    s1 = int((x - (-1.2)) / 1.8 * div)
    s2 = int((v - (-0.07)) / 0.14 * div)
    act = q[s1][s2].index(max(q[s1][s2])) # 方策(greedy)
    obs, rew, term, trunc, info = env.step(act)
    x, v = obs
    s1a = int((x - (-1.2)) / 1.8 * div)
    s2a = int((v - (-0.07)) / 0.14 * div)

    q[s1][s2][act] += 0.8 * (rew + 0.99 * max(q[s1a][s2a]) - q[s1][s2][act])
    
    if term or trunc:
        obs, info = env.reset()
env.close()
