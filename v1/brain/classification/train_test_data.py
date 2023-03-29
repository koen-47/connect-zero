import json
from typing import List

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


class Connect4Dataset(Dataset):
    def __init__(self, csv_file, num_channels=1, transform=None, label_transform=None):
        data = pd.read_csv(csv_file)
        self.data = data[["board_state", "optimal_move", "result"]].copy()
        self.num_channels = num_channels
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        board = torch.tensor(np.array(json.loads(self.data["board"][item]), dtype=np.float32))
        if self.num_channels == 2:
            channel_p1, channel_p2 = split_board_state(self.data["board"][item])
            board = torch.tensor(np.array([channel_p1, channel_p2], dtype=np.float32))
        policy = torch.tensor(np.array(self.data["policy"][item]))
        value = torch.tensor([self.data["value"][item]], dtype=torch.float32)
        return board, policy, value


def process_data(file_path: str):
    raw_data = pd.read_csv(file_path).drop_duplicates()
    raw_data = raw_data.sample(frac=1).reset_index(drop=True)

    filtered_turn_num = pd.DataFrame()
    for i in range(1, 22):
        games = raw_data.loc[raw_data["turn_num"] == i][:125000]
        filtered_turn_num = filtered_turn_num.append(games)
    filtered_turn_num = filtered_turn_num.sample(frac=1).reset_index(drop=True)

    filtered_result = pd.DataFrame()
    for i in range(-1, 2):
        games = filtered_turn_num.loc[filtered_turn_num["result"] == i][:640000]
        filtered_result = filtered_result.append(games)
    filtered_result = filtered_result.sample(frac=1).reset_index(drop=True).drop(["Unnamed: 0"], axis=1)

    processed_data_p1 = filtered_result.loc[filtered_result["player_turn"] == 1]
    processed_data_p2 = filtered_result.loc[filtered_result["player_turn"] == 2]

    processed_data_p1.to_csv("../../data/classification/processed_p1_game_data.csv")
    processed_data_p2.to_csv("../../data/classification/processed_p2_game_data.csv")


def split_train_val_set(csv_path: str):
    dataset = Connect4Dataset(csv_file=csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_data, val_data


def split_board_state(board_str: str):
    board_state = np.array(json.loads(board_str), dtype=np.float32).flatten()
    channel_p1 = np.where(board_state == 2, 0, board_state).reshape((6, 7))
    channel_p2 = np.where(board_state == 1, 0, board_state).reshape((6, 7))
    return channel_p1, channel_p2
