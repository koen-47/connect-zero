import ast
import copy
import csv
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from game.game import Game
from strategies.strategy import AlphaBetaPruningStrategy, RandomStrategy
from brain.classification.train_classifier import Classifier1
from brain.classification.train_test_data import split_train_val_set
from strategies.mcts import MCTS


def execute_episode(num_games: int):
    game = Game()
    wins = [0, 0, 0]
    examples = []
    for i in range(num_games):
        game.reset()

        # if i % 100 == 0:
        #     print(i)

        turn_num = 1
        p1_strategy = MCTS(model=None, player_id=1)
        p2_strategy = MCTS(model=None, player_id=2)
        local_training_data = []
        while not game.is_game_over():
            p1_move_enc = p1_strategy.get_action_probability(game.board.board, 1, temp=0)
            p1_state = copy.deepcopy(game.board.board)
            game.board.drop(1, np.argmax(p1_move_enc))
            local_training_data.append((p1_state, p1_move_enc))

            game_status = game.board.check_win()
            if game_status != -1:
                break

            p2_move_enc = p2_strategy.get_action_probability(game.board.board, 2, temp=0)
            p2_state = copy.deepcopy(game.board.board)
            game.board.drop(2, np.argmax(p2_move_enc))
            local_training_data.append((p2_state, p2_move_enc))
            turn_num += 1

        game_status = game.board.check_win()
        if game_status == 2:
            wins[2] += 1
        elif game_status == 0:
            wins[1] += 1
        elif game_status == 1:
            wins[0] += 1
        for data in local_training_data:
            data = data + (-1 if game_status == 2 else game_status,)
            examples.append(data)
    return examples


def train(model, examples, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sum_loss = 0.0
    count_loss = 0.0
    for epoch in range(num_epochs):
        model.train()

        batch_size = 256
        batch_count = max(1, int(len(examples) / batch_size))
        for i in range(batch_count):
            sample_ids = np.random.randint(len(examples), size=batch_size)
            boards, move, value = list(zip(*[examples[i] for i in sample_ids]))
            boards = torch.FloatTensor(np.array(boards).astype(np.float64)).unsqueeze(dim=1).to(device)
            target_move = torch.FloatTensor(np.array(move)).to(device)
            target_value = torch.FloatTensor(np.array(value).astype(np.float64)).unsqueeze(dim=1).to(device)

            out_move, out_value = model(boards)
            loss_move = criterion1(target_move, out_move)
            loss_value = criterion2(target_value, out_value)
            total_loss = loss_move + loss_value
            sum_loss += total_loss.item()
            count_loss += 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    print(f"Average loss: {sum_loss / count_loss:.3f}")
    return model


def select_action(board, model, device):
    state_2 = torch.FloatTensor(np.array(board.board).astype(np.float64)).to(device). \
        unsqueeze(dim=0).unsqueeze(dim=0)
    available_actions = board.get_valid_moves()
    with torch.no_grad():
        r_actions = model(state_2)[0].to("cpu").squeeze()
        state_action_values = [r_actions[action] for action in available_actions]
        argmax_action = np.argmax(state_action_values)
        greedy_action = available_actions[argmax_action]
        return greedy_action


def select_action_mcts(board, model, device):
    pass


def win_rate_test(model, device):
    game = Game()
    model = model.to(device)

    win_moves_taken_list = []
    win = []
    opponent_strat = AlphaBetaPruningStrategy(player_id=2, depth=3)

    for i in range(100):
        game.reset()
        win_moves_taken = 0

        while not game.is_game_over():
            state_1 = game.board.board
            action = opponent_strat.calculate_move(state_1)
            game.board.drop(1, action)

            if game.board.check_win() == 1:
                break

            action = select_action(game.board, model, device=device)
            game.board.drop(2, action)
            win_moves_taken += 1

            if game.board.check_win() == 2:
                win_moves_taken_list.append(win_moves_taken)
                win.append(1)
                break

    game.reset()
    num_moves_taken = len(win_moves_taken_list) if len(win_moves_taken_list) > 0 else 1
    return sum(win) / 100, sum(win_moves_taken_list) / num_moves_taken


def arena(model_1, model_2, device, num_games=100, win_threshold=0.55):
    game = Game()
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    win_model_1 = 0
    win_model_2 = 0
    num_draws = 0

    halftime = int(num_games - (num_games/2))
    for i in range(halftime):
        game.reset()

        while not game.is_game_over():
            action = select_action(game.board, model_1, device=device)
            game.board.drop(1, action)

            if game.board.check_win() == 1:
                win_model_1 += 1
                break

            action = select_action(game.board, model_2, device=device)
            game.board.drop(2, action)

            if game.board.check_win() == 2:
                win_model_2 += 1
                break
        # print(np.array(game.board.board))

    for i in range(halftime):
        game.reset()

        while not game.is_game_over():
            action = select_action(game.board, model_2, device=device)
            game.board.drop(1, action)

            if game.board.check_win() == 1:
                win_model_2 += 1
                break

            action = select_action(game.board, model_1, device=device)
            game.board.drop(2, action)

            if game.board.check_win() == 2:
                win_model_1 += 1
                break

    # print(wins)
    if (win_model_2 / num_games) > win_threshold:
        print(f"Accepting new model... Win rate: {win_model_2 / num_games}")
        return model_2
    print(f"Rejecting new model... Win rate: {win_model_1 / num_games}")
    return model_1


def load_initial_data(file_path: str):
    initial_data = []
    with open(file_path, newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for data in raw_data[1:]:
            initial_data.append((
                json.loads(data[1]),
                json.loads(data[2]),
                int(data[3])
            ))
    return initial_data


def learn():
    num_iterations = 128
    num_episodes = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Classifier1()

    print("Starting supervised part...")
    initial_data = load_initial_data("../../data/classification/raw_game_data_v2.csv")
    model = train(model, initial_data, num_epochs=10)

    # win_rate, moves_taken = win_rate_test(model, device)
    # print(f"win_rate: {win_rate:.2f}, moves_taken: {moves_taken:.3f}")

    # arena(Classifier1(), model, device=device)

    print("Starting self-play part...")
    for i in range(num_iterations):
        examples = execute_episode(num_episodes)
        random.shuffle(examples)
        model_new = train(model, examples, num_epochs=10)
        model = arena(model, model_new, device=device, num_games=40, win_threshold=0.55)
        torch.save(model.state_dict(), "../../models/saved/dqn_cnn_v2_2.pth")

        # if i % 5 == 0:
        #     win_rate, moves_taken = win_rate_test(model, device)
        #     print(f"win_rate: {win_rate:.3f}, moves_taken: {moves_taken:.3f}")


learn()
