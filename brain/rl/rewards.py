from typing import List


def constant_reward(player_id: int, game_status: int):
    if game_status == 1 and player_id == 1:
        return 1
    elif game_status == 1 and player_id == 2:
        return -1
    elif game_status == 2 and player_id == 1:
        return -1
    elif game_status == 2 and player_id == 2:
        return 1
    return 0.5


def sequence_count_reward(board: List[List[int]]):
    def count_sequences(player_id: int):
        height = len(board)
        width = len(board[0])
        seq_counts = [0, 0, 0]

        def count_sequence_of_length(length: int):
            def is_sequence(x: int, y: int, offset_x: int, offset_y):
                is_seq_bool = True
                for i in range(length):
                    # print(f"cell: {board[x + (offset_x * i)][y + (offset_y * i)]}, x: {x + (offset_x * i)}, y: {y + (
                    # offset_y * i)}")
                    is_seq_bool &= board[x + (offset_x * i)][y + (offset_y * i)] == player_id
                return is_seq_bool

            def count_horizontal_sequences():
                for row in range(height):
                    for col in range(width - length + 1):
                        if is_sequence(row, col, 0, 1):
                            seq_counts[length - 2] += 1

            def count_vertical_sequences():
                for row in range(height - length + 1):
                    for col in range(width):
                        if is_sequence(row, col, 1, 0):
                            seq_counts[length - 2] += 1

            def count_lr_diag_sequences():
                for row in range(height - length + 1):
                    for col in range(width - length + 1):
                        if is_sequence(row, col, 1, 1):
                            seq_counts[length - 2] += 1

            def count_rl_diag_sequences():
                for row in range(height):
                    for col in range(width - length + 1):
                        if is_sequence(row, col, -1, 1):
                            seq_counts[length - 2] += 1

            count_horizontal_sequences()
            count_vertical_sequences()
            count_lr_diag_sequences()
            count_rl_diag_sequences()

        count_sequence_of_length(2)
        count_sequence_of_length(3)
        count_sequence_of_length(4)
        return seq_counts

    seq_counts_p1 = count_sequences(1)
    seq_counts_p2 = count_sequences(2)

    p1_reward = seq_counts_p1[2] * 10000 + seq_counts_p1[1] * 100 + seq_counts_p1[0] * 10
    p2_reward = seq_counts_p2[2] * 10000 + seq_counts_p2[1] * 100 + seq_counts_p2[0] * 10

    return p1_reward - p2_reward
