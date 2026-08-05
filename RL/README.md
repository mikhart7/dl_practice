# RL

Домашние задания курса Practical RL (Yandex School of Data Analysis) — от классического policy search до глубокого RL.

## deep_crossentropy_method.ipynb — Deep Cross-Entropy Method
Policy на базе `MLPClassifier` (sklearn). Обучение на CartPole-v0 (mean reward ≈ 212) и MountainCar-v0 (mean reward ≈ -100 при требовании задания ≥ -150).

## homework_pytorch_main.ipynb — Deep Q-Network
Atari Breakout (`ALE/Breakout-v5`, gymnasium). Препроцессинг кадров (grayscale, resize, стек из 4 кадров), Dueling-архитектура сети, Double DQN, experience replay, target-сеть, epsilon-greedy policy с затуханием.

## hw_continuous_control_pytorch_1.ipynb — Continuous Control
MuJoCo Ant-v4. Реализован TD3 (Twin Delayed DDPG): два critic'а (twin trick), target-сети, replay buffer.
**Результат:** mean reward ≈ 2000 после ~355 000 итераций обучения (требование задания ≥ 1000).

`rl.mp4` — запись поведения обученного агента.
