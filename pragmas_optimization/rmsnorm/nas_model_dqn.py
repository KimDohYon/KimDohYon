import os
import sys
import random
import pickle
import numpy as np
import tensorflow as tf
from collections import deque
from parse_power import (
    load_baseline,
    evaluate_score,
    update_final_results,
    parse_results,
    parse_power_report,
    append_results_with_metrics
)

factors = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768]
num_factors = len(factors)
num_pragmas = 5
num_actions = num_factors ** num_pragmas
input_shape = 2


def decode_action(action):
    return [factors[(action // (num_factors ** i)) % num_factors] for i in range(num_pragmas)]


def generate_rmsnorm_cpp(pragmas):
    with open("rmsnorm_template.cpp", "r", encoding="utf-8", errors="replace") as f:
        template = f.read()
    placeholders = [
        "__SUM_UNROLL__", "__NORM_UNROLL__",
        "__X_BUFF_PARTITION__", "__WEIGHT_BUFF_PARTITION__", "__OUT_BUFF_PARTITION__"
    ]
    for i, ph in enumerate(placeholders):
        template = template.replace(ph, str(pragmas[i]))
    with open("rmsnorm.cpp", "w") as f:
        f.write(template)


def parse_last_result():
    if not os.path.exists("final_results.csv"):
        return None, None, -1
    with open("final_results.csv", "r") as f:
        lines = [line.strip().split(",") for line in f if not line.startswith("sum_")]
        valid = [line for line in lines if len(line) > 11 and float(line[11]) > 0]  # score > 0
        if not valid:
            return None, None, -1
        last = valid[-1]
        interval = float(last[5])
        power = float(last[10])
        score = float(last[11])
        return [interval, power], score, len(valid)


def get_all_used_actions():
    if not os.path.exists("used_actions.txt"):
        return set()
    with open("used_actions.txt", "r") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


class QNetwork(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(num_actions)

    def call(self, x):
        return self.out(self.dense2(self.dense1(x)))


class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.num_actions = num_actions
        self.model = QNetwork(input_shape, num_actions)
        self.model.build((None, input_shape))
        self.target_model = QNetwork(input_shape, num_actions)
        self.target_model.build((None, input_shape))
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.memory = deque(maxlen=1000)
        self.gamma, self.epsilon = 0.99, 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9477  # 💡 Changed from 0.995 to reach 0.05 in ~64 steps
        self.batch_size = 1

        if os.path.exists("dqn_model_weights.weights.h5"):
            self.model.load_weights("dqn_model_weights.weights.h5")
            self.target_model.set_weights(self.model.get_weights())

        if os.path.exists("dqn_memory.pkl"):
            with open("dqn_memory.pkl", "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.memory = data.get("memory", deque(maxlen=1000))
                    self.epsilon = data.get("epsilon", 1.0)
                else:
                    self.memory = data


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q = self.model(np.expand_dims(state, 0))
        return int(tf.argmax(q[0]))

    def store_experience(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            print("[REPLAY] Skipped - not enough memory", file=sys.stderr)
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*minibatch)
        s, s2 = np.array(s), np.array(s2)
        q = self.model(s).numpy()
        q_target = self.target_model(s2).numpy()

        for i in range(self.batch_size):
            q[i][a[i]] = r[i] if d[i] else r[i] + self.gamma * np.max(q_target[i])

        with tf.GradientTape() as tape:
            pred = self.model(s)
            loss = tf.reduce_mean(tf.keras.losses.MSE(q, pred))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            old_eps = self.epsilon
            self.epsilon *= self.epsilon_decay
            print(f"[EPSILON] Decayed from {old_eps:.4f} to {self.epsilon:.4f}", file=sys.stderr)

        self.target_model.set_weights(self.model.get_weights())
        self.model.save_weights("dqn_model_weights.weights.h5")
        with open("dqn_memory.pkl", "wb") as f:
            pickle.dump({"memory": self.memory, "epsilon": self.epsilon}, f)

        print(f"[TRAINING] loss={loss:.6f}, epsilon={self.epsilon:.4f}, memory={len(self.memory)}", file=sys.stderr)
        return float(loss)


if __name__ == "__main__":
    agent = DQNAgent(input_shape=2, num_actions=num_actions)
    prev_state, prev_score, _ = parse_last_result()
    if prev_state is None:
        prev_state, prev_score = [random.uniform(1000, 6000), random.uniform(0.3, 1.2)], 0.0

    a_taken = agent.act(prev_state)
    reward = prev_score if prev_score > 0 else 0.001
    agent.store_experience(prev_state, a_taken, reward, prev_state, False)
    loss = agent.replay()

    used = get_all_used_actions()
    action = agent.act(prev_state)
    while action in used:
        action = random.randint(0, num_actions - 1)
    pragmas = decode_action(action)
    generate_rmsnorm_cpp(pragmas)
    print(",".join(map(str, pragmas)))
    with open("used_actions.txt", "a") as f:
        f.write(f"{action}\n")

    results_fields, interval = parse_results()
    power = parse_power_report()
    base_i, base_p = load_baseline()
    score = evaluate_score(interval, power, base_i, base_p)

    if score > 0.0:
        if loss is None:
            loss = 0.0
        update_final_results(results_fields, power, score, agent.epsilon, loss)
        append_results_with_metrics(pragmas, interval, results_fields, power, score, agent.epsilon, loss)