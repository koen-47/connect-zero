import pandas as pd


def load_data(file_path: str):
    raw_data = pd.read_csv(file_path).drop_duplicates()
    raw_data = raw_data.sample(frac=1).reset_index(drop=True)
    # print(raw_data)
    # print(len(raw_data))
    # print(raw_data["turn_num"].value_counts())

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

    # print(processed_data_p1["player_turn"].value_counts())
    # print(processed_data_p1["turn_num"].value_counts())
    # print(processed_data_p1["optimal_move"].value_counts())
    # print(processed_data_p1["result"].value_counts())

    # print(processed_data_p2["player_turn"].value_counts())
    # print(processed_data_p2["turn_num"].value_counts())
    # print(processed_data_p2["optimal_move"].value_counts())
    # print(processed_data_p2["result"].value_counts())

    # print(filtered_result)

    # filtered_result.to_csv("../../data/classification/processed_game_data.csv")
    processed_data_p1.to_csv("../../data/classification/processed_p1_game_data.csv")
    processed_data_p2.to_csv("../../data/classification/processed_p2_game_data.csv")



load_data("../../data/classification/game_data.csv")
