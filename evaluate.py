import gymnasium as gym
import torch
import time

# main.py에 정의된 QNetwork와 DQNAgent 클래스를 그대로 가져옴.
from main import QNetwork, DQNAgent


def evaluate_agent(env_name="CartPole-v1", model_path="./models/dqn_cartpole_model.pth", num_episodes=5):
    """
    학습된 DQN 에이전트의 성능을 시각적으로 평가하고, 총 보상을 콘솔에 출력함.

    Args:
        env_name (str): 평가할 Gymnasium 환경의 이름.
        model_path (str): 불러올 학습된 모델의 가중치 파일 경로.
        num_episodes (int): 평가를 진행할 에피소드의 수.
    """

    # 시뮬레이션 창을 화면에 띄우기 위해 'human' 렌더 모드로 환경을 생성함.
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 에이전트를 생성함. 평가 시에는 하이퍼파라미터가 중요하지 않으므로 기본값 사용.
    agent = DQNAgent(state_dim, action_dim)

    # 저장된 모델의 가중치(state_dict)를 불러옴.
    # map_location을 통해 CPU에서도 모델을 로드할 수 있도록 설정함.
    try:
        agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없음.")
        print("먼저 main.py를 실행하여 모델을 학습하고 저장해야 함.")
        return

    # 에이전트의 네트워크를 평가 모드(evaluation mode)로 설정함.
    # 이는 드롭아웃이나 배치 정규화 등의 동작을 비활성화하여 일관된 출력을 보장함.
    agent.q_net.eval()

    print(f"학습된 에이전트의 평가를 시작합니다. (총 {num_episodes} 에피소드)")

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 렌더링 속도를 조절하여 사람이 볼 수 있도록 약간의 지연을 줌.
            time.sleep(0.01)

            # 에이전트가 행동을 선택함.
            # eval_mode=True로 설정하여 탐험(epsilon-greedy)을 하지 않고,
            # 학습된 정책에 따라 최적의 행동만 선택하도록 함.
            action = agent.select_action(state, eval_mode=True)

            # 선택한 행동을 환경에서 실행함.
            next_state, reward, done, _, _ = env.step(action)

            state = next_state
            total_reward += reward

            # 콘솔에 현재 보상 점수 실시간 출력
            print(f"\rEpisode {ep + 1} | Current Reward: {total_reward:.0f}", end="")
        print()

    env.close()
    print("\n평가 완료.")


if __name__ == "__main__":
    evaluate_agent()
