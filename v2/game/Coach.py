import copy
import multiprocessing

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from random import shuffle
from multiprocessing import Process, Manager, Pool

from v2.game.Game import Game
from v2.game.Player import Player
from v2.game.Arena import Arena
from v2.strategy.AlphaZeroStrategy import AlphaZeroStrategy
# from models.keras import DQN1
# from models.keras.DQN1 import Connect4NNet
from v2.models.pytorch.ResNet import ResNet
from v2.strategy.AlphaZeroStrategy import MCTS


class Coach:
    def __init__(self, game, nnet, num_its=100, num_eps=100, temp_threshold=15):
        self.game = game
        self.nnet = nnet
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pnet = self.nnet.__class__(128, 20)
        self.mcts = MCTS(self.game, self.nnet, self.device)
        self.trainExamplesHistory = []
        self.num_its = num_its
        self.num_eps = num_eps
        self.temp_threshold = temp_threshold

    def execute_episode(self):
        trainExamples = []
        board = self.game.get_init_board()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.get_canonical_form(board, self.curPlayer)
            temp = int(episodeStep < self.temp_threshold)

            pi = self.mcts.get_action_prob(canonicalBoard, device=self.device, temp=temp)
            sym = self.game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.get_next_state(board, self.curPlayer, action)

            r = self.game.get_game_ended(board, self.curPlayer)

            if r != 0:
                print(self.game.display(board))
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
            iterationTrainExamples = []
            pool = Pool(processes=num_proc)
            for examples in pool.starmap(func=self.parallel, iterable=[(self.num_eps, num_proc,) for _ in
                                                                       range(int(num_proc))]):
                for example in examples:
                    iterationTrainExamples.append(example)


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

            # print(iterationTrainExamples)
            print(f"Number of training examples: {len(iterationTrainExamples)}")
            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            pmcts = MCTS(self.game, self.pnet, self.device)

            self.nnet = self.nnet.train_on_examples(trainExamples, lr=0.0001)
            nmcts = MCTS(self.game, self.nnet, self.device)

            player1 = Player(1, strategy=AlphaZeroStrategy(mcts=pmcts))
            player2 = Player(-1, strategy=AlphaZeroStrategy(mcts=nmcts))
            arena = Arena(player1, player2)
            pwins, nwins, draws = arena.play_games(num_games)

            print(f"Results (pwins/nwins/draws) => ({pwins}, {nwins}, {draws})")
            if nwins / num_games >= win_threshold:
                print("Accepting new model!")
                self.pnet = copy.deepcopy(self.nnet)
            else:
                print("Rejecting new model")
                self.nnet = copy.deepcopy(self.pnet)
            torch.save(self.nnet.state_dict(), "../models/saved/resnet_1.pth")


# print(torch.cuda.is_available())

if __name__ == '__main__':
    g = Game()
    nnet = ResNet(num_channels=128, num_res_blocks=20)
    coach = Coach(game=g, nnet=nnet, num_its=80, num_eps=24)
    coach.learn(num_games=10)

# episode = coach.execute_episode()
# nnet.train_on_examples(episode)
# print(episode)
