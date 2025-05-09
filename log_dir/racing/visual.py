import matplotlib.pyplot as plt
import csv

def plot_average_reward(log_file_path, output_path='new_average_reward_per_episode.png'):

    episodes = []
    rewards = []

    try:
        with open(log_file_path, 'r') as file:
            reader = csv.DictReader(file)  # Используем csv.DictReader для удобства
            for row in reader:
                episode = int(row['episode'])
                reward = float(row['reward'])
                episodes.append(episode)
                rewards.append(reward)
    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
        return
    except KeyError:
        print("Error in format")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Номер эпизода', fontsize=12)
    plt.ylabel('Средняя награда', fontsize=12)
    plt.title('Средняя награда по эпизодам', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.show()

if __name__ == '__main__':
    log_file = 'PPO_log.csv'
    plot_average_reward(log_file)
    print("Success!")


# import matplotlib.pyplot as plt
# import csv
# import numpy as np

# def moving_average(data, window_size):
#     cumsum = np.cumsum(np.insert(data, 0, 0))
#     return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# def plot_average_reward_smoothed(log_file_path, output_path='average_reward_per_episode_smoothed.png', window=100):

#     episodes = []
#     rewards = []

#     try:
#         with open(log_file_path, 'r') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 episode = int(row['episode'])
#                 reward = float(row['reward'])
#                 episodes.append(episode)
#                 rewards.append(reward)
#     except FileNotFoundError:
#         print(f"Error: File not found: {log_file_path}")
#         return
#     except KeyError:
#         print("Error in format")
#         return

#     smoothed_rewards = moving_average(rewards, window)
#     smoothed_episodes = episodes[window - 1:]  # Сдвигаем номера эпизодов

#     plt.figure(figsize=(10, 5))
#     plt.plot(smoothed_episodes, smoothed_rewards, linestyle='-', color='royalblue', linewidth=2)
#     plt.xlabel('Номер эпизода', fontsize=12)
#     plt.ylabel('Средняя награда (скользящее среднее)', fontsize=12)
#     plt.title('Средняя награда по эпизодам', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.tight_layout()

#     plt.savefig(output_path)
#     plt.show()

# if __name__ == '__main__':
#     log_file = 'PPO_log.csv'
#     plot_average_reward_smoothed(log_file, window=200)