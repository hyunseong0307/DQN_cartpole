import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# --- 모델 저장을 위한 디렉토리 생성 ---
if not os.path.exists('models'):
    os.makedirs('models')


###################################
# Q-Network 정의
# 상태(State)를 입력으로 받아 각 행동(Action)의 가치(Q-Value)를 출력하는 신경망임.
###################################
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        # super()는 nn.Module 클래스의 초기화자를 호출함.
        super(QNetwork, self).__init__()

        # 신경망의 레이어를 정의함.
        # state_dim: 입력 차원 (CartPole의 경우 4)
        # action_dim: 출력 차원 (CartPole의 경우 2, 왼쪽/오른쪽)
        # hidden_size: 은닉층의 노드 수
        self.fc1 = nn.Linear(state_dim, hidden_size)  # 첫 번째 완전 연결 레이어
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 두 번째 완전 연결 레이어
        self.fc3 = nn.Linear(hidden_size, action_dim)  # 출력 레이어

    def forward(self, x):
        # 입력(x)으로부터 순전파를 수행하여 Q-값을 계산함.
        # ReLU 활성화 함수를 사용하여 비선형성을 추가함.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 마지막 레이어는 활성화 함수를 거치지 않은 Q-값을 그대로 출력함.
        q_values = self.fc3(x)
        return q_values


###################################
# 경험 리플레이 버퍼 (Experience Replay Buffer)
# 에이전트의 경험(state, action, reward, next_state, done)을 저장하고,
# 학습 시 무작위로 샘플링하여 신경망에 제공함.
# 이를 통해 데이터 간의 시간적 상관관계를 깨고 학습을 안정화시킴.
###################################
class ReplayBuffer:
    def __init__(self, capacity=50000):
        # deque는 양쪽 끝에서 데이터를 추가하거나 제거할 수 있는 자료구조임.
        # maxlen을 설정하여 버퍼의 최대 크기를 제한함.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 버퍼에 경험 튜플을 추가함.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 버퍼에서 batch_size만큼의 경험을 무작위로 샘플링함.
        batch = random.sample(self.buffer, batch_size)
        # 샘플링된 배치 데이터를 각 요소별로 분리하여 반환함.
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        # 버퍼에 저장된 경험의 수를 반환함.
        return len(self.buffer)


###################################
# DQN 에이전트
# Q-Network와 Replay Buffer를 사용하여 환경과 상호작용하고 학습을 수행함.
###################################
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=5e-4, tau=500, batch_size=128, replay_capacity=50000):
        self.action_dim = action_dim  # 행동의 차원
        self.gamma = gamma  # 할인율 (미래 보상의 가치를 현재 가치로 환산)
        self.lr = lr  # 학습률 (Learning Rate)
        self.tau = tau  # 타겟 네트워크 업데이트 주기
        self.batch_size = batch_size  # 학습 시 사용할 미니배치 크기

        # GPU 사용 가능 여부에 따라 device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 메인 네트워크(q_net)와 타겟 네트워크(target_net)를 생성함.
        # 타겟 네트워크는 학습을 안정화시키는 역할을 함.
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        # 타겟 네트워크의 가중치를 메인 네트워크와 동일하게 초기화함.
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Adam 옵티마이저를 사용하여 q_net의 파라미터를 최적화함.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        # 리플레이 버퍼를 생성함.
        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.total_steps = 0  # 총 스텝 수
        self.epsilon = 1.0  # ε-greedy 정책의 초기 epsilon 값 (탐험 확률)
        self.epsilon_decay = 0.998  # epsilon 감소율
        self.epsilon_min = 0.01  # epsilon의 최소값

    def select_action(self, state, eval_mode=False):
        # ε-greedy 정책에 따라 행동을 선택함.
        # eval_mode=True일 경우, 탐험(무작위 행동)을 하지 않고 오직 활용(최적 행동)만 함.
        if eval_mode:
            self.epsilon = 0

        if random.random() < self.epsilon:
            # 확률 epsilon으로 무작위 행동을 선택 (탐험, Exploration)
            return random.randrange(self.action_dim)
        else:
            # 확률 1-epsilon으로 Q-값이 가장 높은 행동을 선택 (활용, Exploitation)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.q_net(state_t)
            action = q_vals.argmax(dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        # 경험을 리플레이 버퍼에 저장함.
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self):
        # 리플레이 버퍼에 충분한 데이터가 쌓이기 전까지는 학습을 시작하지 않음.
        if len(self.replay_buffer) < self.batch_size * 10:
            return

        # 버퍼에서 미니배치를 샘플링함.
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 샘플링된 데이터를 PyTorch 텐서로 변환하고 device로 보냄.
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # --- 손실(Loss) 계산 ---
        # 1. 현재 상태(states_t)에서 실제 취한 행동(actions_t)의 Q-값을 계산함.
        #    gather(1, actions_t)는 각 상태에 대한 Q-값들 중, 실제 선택한 행동의 Q-값만 가져옴.
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # 2. 타겟 Q-값을 계산함. 타겟 네트워크는 가중치가 고정되어 있어 학습을 안정시킴.
        with torch.no_grad():  # 타겟 네트워크는 그래디언트 계산이 필요 없음.
            # 다음 상태(next_states_t)에서 가장 큰 Q-값을 타겟 네트워크로부터 계산함.
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            # 벨만 기대 방정식에 따라 타겟 Q-값을 계산함.
            # done=True인 경우, 미래 가치가 없으므로 보상(reward)만 사용함.
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        # 3. 손실 함수로 Mean Squared Error(MSE)를 사용하여 두 Q-값의 차이를 계산함.
        loss = F.mse_loss(q_values, target_q)

        # --- 네트워크 업데이트 ---
        self.optimizer.zero_grad()  # 이전 그래디언트를 초기화함.
        loss.backward()  # 역전파를 통해 그래디언트를 계산함.
        self.optimizer.step()  # 옵티마이저를 통해 메인 네트워크의 가중치를 업데이트함.

        # --- 타겟 네트워크 업데이트 ---
        # 일정 스텝(tau)마다 메인 네트워크의 가중치를 타겟 네트워크로 복사함. (Hard Update)
        if self.total_steps % self.tau == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon 값을 점진적으로 감소시켜 탐험의 비중을 줄여나감.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return loss.item()  # loss 값을 반환하여 학습 과정을 모니터링할 수 있음.


###################################
# 학습 실행 함수
###################################
def train_dqn(env_name="CartPole-v1", max_episodes=1000):
    # Gymnasium 환경을 생성함.
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN 에이전트를 생성함.
    agent = DQNAgent(state_dim, action_dim)

    # 최근 100개 에피소드의 보상을 저장하기 위한 deque
    reward_history = deque(maxlen=100)
    best_avg_reward = -np.inf  # 최고 평균 보상을 기록하기 위한 변수

    print("학습을 시작합니다...")
    for ep in range(max_episodes):
        state, _ = env.reset()  # 환경 초기화
        total_reward = 0
        done = False

        while not done:
            # 에이전트가 행동을 선택
            action = agent.select_action(state)
            # 선택한 행동을 환경에서 실행하고 다음 상태, 보상, 종료 여부 등을 받음
            next_state, reward, done, _, _ = env.step(action)

            # 경험을 리플레이 버퍼에 저장
            agent.store_transition(state, action, reward, next_state, done)
            # 에이전트의 네트워크를 업데이트 (학습)
            agent.update()

            state = next_state
            total_reward += reward

        # 에피소드 종료 후 보상 기록
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history)

        # 20 에피소드마다 학습 진행 상황 출력
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1} | 최근 100개 평균 보상: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        # 최고 평균 보상을 갱신할 때마다 모델의 가중치를 파일로 저장
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(agent.q_net.state_dict(), "./models/dqn_cartpole_model.pth")
            print(f"Episode {ep + 1} | 최고 평균 보상 갱신: {best_avg_reward:.2f}. 모델을 저장합니다.")

        # 학습 종료 조건: 최근 100개 에피소드의 평균 보상이 495점 이상일 때
        # CartPole-v1 환경은 500점이 만점이므로, 평균 495점 이상이면 충분히 학습된 것으로 간주함.
        if avg_reward >= 495:
            print(f"\n환경 해결! Episode {ep + 1}에서 학습을 조기 종료합니다.")
            break

    env.close()
    print("학습 완료.")


if __name__ == "__main__":
    train_dqn()
