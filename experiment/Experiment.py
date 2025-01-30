import json
import re

import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches

from game.Game import Game
from game.Player import Player
from brain.Evaluator import Evaluator
from strategy.AlphaBetaPruningStrategy import AlphaBetaPruningStrategy
from strategy.RandomStrategy import RandomStrategy
from strategy.AlphaZeroStrategy import AlphaZeroStrategy
from brain.MCTS import MCTS
from models.DualResidualNetwork import DualResidualNetwork
from logs.Logger import Logger


class Experiment:
    def __init__(self, model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ = DualResidualNetwork(num_channels=128, num_res_blocks=5).to(device)
        model_.load_state_dict(torch.load(model))
        model_.eval()
        self.__mcts = MCTS(game=Game(), model=model_, device=device, num_sims=500, c_puct=2., dir_e=0)
        self.__logger = Logger()

    def run(self, n_games, log_losses=True):
        strategies = {"random": RandomStrategy()}
        for i in range(2, 11):
            strategies[f"alphabeta_{i}"] = AlphaBetaPruningStrategy(depth=i)

        for name, strategy_2 in strategies.items():
            player_1 = Player(1, strategy=AlphaZeroStrategy(mcts=self.__mcts))
            player_2 = Player(-1, strategy=strategy_2)
            evaluator = Evaluator(player_1, player_2)
            results, states = evaluator.play_games(n_games, return_states=True)
            n_player_2_wins, n_draws, n_player_1_wins = results
            win_rate = n_player_1_wins / sum(results)
            print(win_rate, results)

            if log_losses:
                self.__logger.set_log_experiment_file(name, f"../experiment/logs/recent/experiment_{name}")
                self.__logger.log(f"Win rate: {win_rate}", to_experiment=True)
                self.__logger.log(f"Wins: {n_player_1_wins}. Draws: {n_draws}. Losses: {n_player_2_wins}.\n",
                                  to_experiment=True)

                self.__logger.log("Losses")
                losses = [(half, state) for half, state, result in states if result == -1]
                for i, (half, loss) in enumerate(losses):
                    for j, (state, player_id, action, policy) in enumerate(loss):
                        self.__logger.log(f"Half: {half}. Loss: {i + 1}. Turn {j + 1} (player: {player_id})",
                                          to_experiment=True)
                        self.__logger.log(f"Action: {action}. Policy: {policy}", to_experiment=True)
                        self.__logger.log(f"{Game().display(state)}", to_experiment=True)

    def plot_result_curves_v1(self, path="./logs/recent", dark_mode=False):
        results_per_iteration = self.__parse_log_summary_file(path)
        value_losses = [result["value_loss"] for result in results_per_iteration]
        policy_accuracies = [result["policy_accuracy"] for result in results_per_iteration]
        model_acceptances = [result["is_accepted"] for result in results_per_iteration]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        color = "#0d1117" if not dark_mode else "#F0F6FC"

        iterations = range(1, len(results_per_iteration) + 1)
        value_loss_line, = axes[0].plot(iterations, value_losses, label="Value loss", color="#459abd", alpha=0.6)
        for i, (x, y) in enumerate(zip(iterations, value_losses)):
            marker = r"$\checkmark$" if model_acceptances[i] else r"$\times$"
            marker_color = "#43b97f" if model_acceptances[i] else "#ff4747"
            axes[0].scatter(x - 0.1, y, marker=marker, color=marker_color, s=100, zorder=2)

        policy_acc_line, = axes[1].plot(iterations, policy_accuracies, label="Policy accuracy", color="#459abd",
                                        alpha=0.6)
        for i, (x, y) in enumerate(zip(iterations, policy_accuracies)):
            marker = r"$\checkmark$" if model_acceptances[i] else r"$\times$"
            marker_color = "#43b97f" if model_acceptances[i] else "#ff4747"
            axes[1].scatter(x - 0.1, y, marker=marker, color=marker_color, s=100, zorder=2)

        checkmark_legend = mlines.Line2D([], [], color="#43b97f", marker=r"$\checkmark$", linestyle="None",
                                         markersize=10, label="Model accepted")
        cross_legend = mlines.Line2D([], [], color="#ff4747", marker=r"$\times$", linestyle="None",
                                     markersize=10, label="Model rejected")

        for ax in axes:
            ax.spines["bottom"].set_color(color)
            ax.spines["left"].set_color(color)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.xaxis.label.set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis="x", colors=color)
            ax.tick_params(axis="y", colors=color)
            ax.grid(False)
            ax.set_xlabel("Iteration")

        axes[0].legend(handles=[value_loss_line, checkmark_legend, cross_legend], frameon=False, labelcolor=color)
        axes[1].legend(handles=[policy_acc_line, checkmark_legend, cross_legend], frameon=False, labelcolor=color)
        axes[0].set_ylabel("MSE Loss")
        axes[1].set_ylabel("Accuracy (%)")

        plt.tight_layout()
        plt.show()

    def plot_result_curves_v2(self, path="./logs/recent", dark_mode=False):
        results_per_iteration = self.__parse_log_summary_file(path)
        value_losses = [result["value_loss"] for result in results_per_iteration]
        policy_accuracies = [result["policy_accuracy"] for result in results_per_iteration]
        model_acceptances = [result["is_accepted"] for result in results_per_iteration]
        win_rates_random = [result["win_rate_random"] for result in results_per_iteration]
        win_rates_alpha_beta = [result["win_rate_alpha_beta"] for result in results_per_iteration]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        color = "#0d1117" if not dark_mode else "#F0F6FC"
        patch_width = 1.

        iterations = range(1, len(results_per_iteration) + 1)
        value_loss_line, = axes[0].plot(iterations, value_losses, label="Value loss", color="#459abd", marker="o")
        value_y_min, value_y_max = axes[0].get_ylim()
        for i, (x, y) in enumerate(zip(iterations, value_losses)):
            patch_color = "#43b97f" if model_acceptances[i] else "#ff4747"
            patch = patches.Rectangle((x - patch_width / 2, value_y_min), patch_width, value_y_max - value_y_min,
                                      color=patch_color, alpha=0.3, zorder=0, linewidth=0, edgecolor=None)
            axes[0].add_patch(patch)

        policy_acc_line, = axes[1].plot(iterations, policy_accuracies, label="Policy accuracy", color="#459abd",
                                        marker="o")
        policy_y_min, policy_y_max = axes[1].get_ylim()
        for i, (x, y) in enumerate(zip(iterations, policy_accuracies)):
            patch_color = "#43b97f" if model_acceptances[i] else "#ff4747"
            patch = patches.Rectangle((x - patch_width / 2, policy_y_min), patch_width, policy_y_max - policy_y_min,
                                      color=patch_color, alpha=0.3, zorder=0, linewidth=0, edgecolor=None)
            axes[1].add_patch(patch)

        win_rates_random_line, = axes[2].plot(iterations, win_rates_random, label="Random", color="#459abd", marker="o")
        win_rates_alpha_beta_line, = axes[2].plot(iterations, win_rates_alpha_beta, label="Alpha-beta (depth: 5)",
                                                  color="#d88a1f", marker="o")
        win_rates_y_min, win_rates_y_max = axes[2].get_ylim()
        for i, (x, y) in enumerate(zip(iterations, policy_accuracies)):
            patch_color = "#43b97f" if model_acceptances[i] else "#ff4747"
            patch = patches.Rectangle((x - patch_width / 2, win_rates_y_min), patch_width, win_rates_y_max - win_rates_y_min,
                                      color=patch_color, alpha=0.3, zorder=0, linewidth=0, edgecolor=None)
            axes[2].add_patch(patch)

        accepted_patch = patches.Patch(color="#43b97f", alpha=0.3, label="Model accepted")
        rejected_patch = patches.Patch(color="#ff4747", alpha=0.3, label="Model rejected")

        for ax in axes:
            ax.spines["bottom"].set_color(color)
            ax.spines["left"].set_color(color)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.xaxis.label.set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis="x", colors=color)
            ax.tick_params(axis="y", colors=color)
            ax.grid(False)
            ax.set_xlabel("Iteration")

        axes[0].legend(handles=[value_loss_line, accepted_patch, rejected_patch], frameon=False, labelcolor=color,
                       loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_ylim(value_y_min, value_y_max)

        axes[1].legend(handles=[policy_acc_line, accepted_patch, rejected_patch], frameon=False, labelcolor=color,
                       loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(policy_y_min, policy_y_max)

        axes[2].legend(handles=[win_rates_random_line, win_rates_alpha_beta_line, accepted_patch, rejected_patch],
                       frameon=False, labelcolor=color, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
        axes[2].set_ylabel("Win rate (%)")
        axes[2].set_ylim(win_rates_y_min, win_rates_y_max)

        plt.tight_layout()
        plt.savefig(f"./experiment/plots/loss_accuracy_winrate_curves_{'dark' if dark_mode else 'light'}.png",
                    transparent=True)
        plt.show()

    def __parse_log_summary_file(self, path="./logs/recent"):
        with open(f"{path}/log_summary", "r") as file:
            results_per_iteration, current_iteration_log = [], ""
            for line in file:
                if line == "\n":
                    losses = re.search("Epoch: 10.*?Value loss: (\d+\.\d+).*?Policy accuracy: (\d+\.\d+)",
                                       current_iteration_log)
                    value_loss = float(losses.group(1))
                    policy_accuracy = float(losses.group(2))
                    is_accepted = bool(re.search("Accepting new model...", current_iteration_log) is not None)

                    win_rate_random = float(re.search("Win rate \(random\): (\d+\.\d+)", current_iteration_log).group(1))
                    win_rate_alpha_beta = float(re.search("Win rate \(alpha-beta pruning with depth 5\): (\d+\.\d+)",
                                                current_iteration_log).group(1))
                    results_per_iteration.append({
                        "value_loss": value_loss,
                        "policy_accuracy": round(policy_accuracy * 100, 1),
                        "is_accepted": is_accepted,
                        "win_rate_random": round(win_rate_random * 100),
                        "win_rate_alpha_beta": round(win_rate_alpha_beta * 100)
                    })
                    current_iteration_log = ""
                else:
                    current_iteration_log += f", {line.strip()}"
            return results_per_iteration
