import copy
import csv
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from game.game import Game
from brain.classification.train_classifier import Classifier1
from mcts import MCTS
from models.cnn.cnn_dqn_3 import DQN_CNN_3


def execute_episode(num_games: int, model):
    game = Game()
    examples = []

    sum_moves_taken = 0
    for i in tqdm(range(num_games)):
        game.reset()

        turn_num = 1
        p1_strategy = MCTS(model=model, player_id=1)
        p2_strategy = MCTS(model=model, player_id=2)
        local_training_data = []
        while not game.is_game_over():
            temp = int(turn_num <= 11)
            p1_state = copy.deepcopy(game.board.board)
            p1_move_enc = p1_strategy.get_action_probability(p1_state, 1, temp=temp, e=0.25)
            game.board.drop(1, np.argmax(p1_move_enc))
            local_training_data.append((p1_state, p1_move_enc))

            game_status = game.board.check_win()
            if game_status != -1:
                break

            p2_state = copy.deepcopy(game.board.board)
            p2_move_enc = p2_strategy.get_action_probability(p2_state, 2, temp=temp, e=0.25)
            game.board.drop(2, np.argmax(p2_move_enc))
            local_training_data.append((p2_state, p2_move_enc))
            turn_num += 1

        if i % 1 == 0:
            print(f"\nSAMPLE FINISHED GAME:\n {np.array(game.board.board)}")

        sum_moves_taken += turn_num
        game_status = game.board.check_win()
        for data in local_training_data:
            data = data + (-1 if game_status == 2 else game_status,)
            # print(data)
            examples.append(data)
    print(f"FINISHED GENERATING SELF-PLAY EXAMPLES")
    print(f"    - Avg. turns per game: {sum_moves_taken / num_games}")
    return examples


def train(model, examples, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()

        sum_total_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0
        sum_policy_acc = 0.0
        total_policy_acc = 0

        batch_size = 64
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

            sum_total_loss += total_loss.item()
            sum_policy_loss += loss_move.item()
            sum_value_loss += loss_value.item()
            total_policy_acc += out_move.size(0)
            sum_policy_acc += (torch.sum(torch.argmax(target_move, dim=1) == torch.argmax(out_move, dim=1))).item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print(f"  EPOCH {epoch+1}) "
              f"AVG. TOTAL LOSS: {sum_total_loss / batch_count:.3f}, "
              f"AVG. VALUE LOSS: {sum_value_loss / batch_count:.3f}, "
              f"AVG. POLICY LOSS: {sum_policy_loss / batch_count:.3f}, "
              f"AVG. POLICY ACC.: {sum_policy_acc / total_policy_acc:.3f}")
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


def arena(model_1, model_2, device, num_games=100, win_threshold=0.55):
    print("PITTING MODELS AGAINST EACH OTHER")

    game = Game()
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    win_model_1 = 0
    win_model_2 = 0
    num_draws = 0

    halftime = int(num_games - (num_games / 2))
    mcts_model_1 = MCTS(model=model_1, player_id=1)
    mcts_model_2 = MCTS(model=model_2, player_id=2)

    print("STARTING FIRST HALF")
    for i in tqdm(range(halftime)):
        game.reset()

        while not game.is_game_over():
            action = mcts_model_1.get_action_probability(game.board.board, player_id=1, temp=0, e=0.25)
            game.board.drop(1, np.argmax(action))

            game_status = game.board.check_win()
            if game_status == 1:
                win_model_1 += 1
                break
            if game_status == 0:
                num_draws += 1
                break

            action = mcts_model_2.get_action_probability(game.board.board, player_id=2, temp=0, e=0.25)
            game.board.drop(2, np.argmax(action))

            game_status = game.board.check_win()
            if game_status == 2:
                win_model_2 += 1
                break
            if game_status == 0:
                num_draws += 1
                break

        if i % 1 == 0:
            print(f"{np.array(game.board.board)}")


    print("STARTING SECOND HALF")
    for i in tqdm(range(halftime)):
        game.reset()

        while not game.is_game_over():
            action = mcts_model_2.get_action_probability(game.board.board, player_id=1, temp=0, e=0.25)
            game.board.drop(1, np.argmax(action))

            game_status = game.board.check_win()
            if game_status == 1:
                win_model_2 += 1
                break
            if game_status == 0:
                num_draws += 1
                break

            action = mcts_model_1.get_action_probability(game.board.board, player_id=2, temp=0, e=0.25)
            game.board.drop(2, np.argmax(action))

            game_status = game.board.check_win()
            if game_status == 2:
                win_model_1 += 1
                break
            if game_status == 0:
                num_draws += 1
                break

        if i % 1 == 0:
            print(f"{np.array(game.board.board)}")

    # print(wins)
    win_rate_model_1 = win_model_1 / num_games
    win_rate_model_2 = win_model_2 / num_games
    print(f"RESULTS => M1_WINS/DRAWS/M2_WINS: {win_rate_model_1}/{num_draws / num_games}/{win_rate_model_2}")
    if win_rate_model_2 >= win_threshold:
        print(f"ACCEPTING NEW MODEL")
        return model_2
    print(f"REJECTING NEW MODEL")
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
    print(f"DEVICE: {device}")

    model = DQN_CNN_3(num_channels=256, num_res_blocks=20, kernel_size=(3, 3), padding=1).to(device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in Classifier1().parameters() if p.requires_grad))

    # print("INITIATING SUPERVISED LEARNING")
    # initial_data = load_initial_data("../../data/classification/raw_game_data_v2.csv")
    # model = train(model, initial_data, num_epochs=10)

    # model = arena(DQN_CNN_3(), model, device=device)

    print("INITIATING SELF-PLAY")
    for i in range(1, num_iterations + 1):
        print(f"ITERATION: {i}")
        examples = execute_episode(num_episodes, model=model)
        random.shuffle(examples)
        model_new = train(model, examples, num_epochs=10)
        # model_new = train(copy.deepcopy(model), examples, num_epochs=10)
        model = arena(model, model_new, device=device, num_games=40, win_threshold=0.55)
        torch.save(model.state_dict(), "../../models/saved/dqn_cnn_v2_6.pth")

        # REMEMBER
        # removed copy.deepcopy() from model for var model_new (line 243, 244)
        # changed cpuct from 1.0 to 4.0
        # changed dirichlet noise from 0.3 to 1.0
        # suggestion: decrease lr

learn()
