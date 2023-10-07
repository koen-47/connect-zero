import copy
import multiprocessing

import torch
from tqdm import tqdm
from random import shuffle

from v2.game.Game import Game
from v2.game.Player import Player
from v2.brain.Arena import Arena
from v2.strategy.AlphaZeroStrategyV2 import AlphaZeroStrategyV2 as AlphaZeroStrategy
# from models.keras import DQN1
# from models.keras.DQN1 import Connect4NNet
from v2.models.pytorch.DualResidualNetwork import DualResidualNetwork
# from v2.strategy.AlphaZeroStrategyV2 import MCTS
from v2.brain.MCTS import MCTS
from v2.logs.Logger import Logger


class Coach:
    def __init__(self, game, num_its=100, num_eps=100, temp_threshold=15):
        self.game = game
        self.nnet = DualResidualNetwork(num_channels=128, num_res_blocks=5)
        self.pnet = copy.deepcopy(self.nnet)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mcts = MCTS(self.game, self.nnet, self.device)
        self.trainExamplesHistory = []
        self.num_its = num_its
        self.num_eps = num_eps
        self.temp_threshold = temp_threshold
        self.logger = Logger()

    def execute_episode(self):
        trainExamples = []
        board = self.game.get_init_board()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.get_canonical_form(board, self.curPlayer)
            temp = int(episodeStep < self.temp_threshold)

            action, pi = self.mcts.get_action_prob(canonicalBoard, device=self.device)
            sym = self.game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            board, self.curPlayer = self.game.get_next_state(board, self.curPlayer, action)

            r = self.game.get_game_ended(board, self.curPlayer)

            if r != 0:
                print(self.game.display(board))
                print(r)
                print(self.curPlayer)
                self.logger.__log_iteration(self.game.display(board, color=False))
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def parallel(self, num_eps, num_proc):
        examples = []
        for _ in tqdm(range(int(num_eps / num_proc)),
                      desc=f"Self Play (proc_id = {multiprocessing.current_process().name.split('-')[-1]})"):
            self.mcts = MCTS(self.game, self.nnet, self.device)  # reset search tree
            examples += self.execute_episode()
        return examples

    def learn(self, num_games=40, win_threshold=0.55, num_proc=4):
        for i in range(1, self.num_its + 1):
            self.logger.set_log_iteration_file(num=i, file=f"../logs/recent/log_it_{i}")
            self.logger.log_both(f"Iteration {i}")
            # LESSON: TRY WITHOUT POOL, BUT WITH MANAGER (THE FIRST, OG APPROACH) (i.e., the code that is commented out)
            iterationTrainExamples = []
            # pool = Pool(processes=num_proc)
            # for examples in pool.starmap(func=self.parallel, iterable=[(self.num_eps, num_proc,) for _ in
            #                                                            range(int(num_proc))]):
            #     for example in examples:
            #         iterationTrainExamples.append(example)

            # print(iterationTrainExamples)
            # manager = Manager()
            # iterationTrainExamples = manager.list()
            # jobs = []
            # for _ in range(num_proc):
            #     process = Process(target=self.parallel, args=(iterationTrainExamples, 3))
            #     jobs.append(process)
            #     process.start()
            #
            # for j in jobs:
            #     j.join()

            for i in tqdm(range(int(self.num_eps)), desc=f"Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.device)  # reset search tree
                self.logger.__log_iteration(f"(Self Play) Episode {i + 1}")
                iterationTrainExamples += self.execute_episode()
                # print(iterationTrainExamples)
                # print(self.curPlayer)

            # print(iterationTrainExamples)
            print(f"Number of training examples: {len(iterationTrainExamples)}")
            self.logger.__log_summary(f"(Self Play) Number of training examples: {len(iterationTrainExamples)}")
            self.trainExamplesHistory.append(iterationTrainExamples)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            pmcts = MCTS(self.game, self.pnet, self.device)

            self.nnet = self.nnet.train_on_examples(trainExamples, lr=0.0001, logger=self.logger)
            nmcts = MCTS(self.game, self.nnet, self.device)

            player1 = Player(1, strategy=AlphaZeroStrategy(mcts=pmcts))
            player2 = Player(-1, strategy=AlphaZeroStrategy(mcts=nmcts))
            arena = Arena(player1, player2, logger=self.logger)
            pwins, nwins, draws = arena.play_games(num_games)

            self.logger.__log_iteration(f"(Arena) Results (pwins/nwins/draws) => ({pwins}, {nwins}, {draws})")
            self.logger.__log_summary(f"(Arena) Results (pwins/nwins/draws) => ({pwins}, {nwins}, {draws})")
            print(f"Results (pwins/nwins/draws) => ({pwins}, {nwins}, {draws})")
            if nwins / num_games >= win_threshold:
                self.logger.__log_iteration("(Arena) Accepting new model.")
                self.logger.__log_summary("(Arena) Accepting new model.")
                print("Accepting new model!")
                self.pnet = copy.deepcopy(self.nnet)
            else:
                self.logger.__log_iteration("(Arena) Rejecting new model.")
                self.logger.__log_summary("(Arena) Rejecting new model.")
                print("Rejecting new model")
                self.nnet = copy.deepcopy(self.pnet)
            torch.save(self.nnet.state_dict(), "../models/saved/resnet_2.pth")
            self.logger.log_both("\n")


# print(torch.cuda.is_available())

if __name__ == '__main__':
    g = Game()
    coach = Coach(game=g, num_its=1, num_eps=1)
    coach.learn(num_games=40, num_proc=1)

# episode = coach.execute_episode()
# nnet.train_on_examples(episode)
# print(episode)
