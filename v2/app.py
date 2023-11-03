import flask
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin

import torch

from game.Game import Game
from brain.MCTS import MCTS
from models.pytorch.DualResidualNetwork import DualResidualNetwork
from strategy.AlphaZeroStrategyV2 import AlphaZeroStrategyV2

app = Flask(__name__)
CORS(app)


@app.route("/connect-zero/predict", methods=["POST"])
@cross_origin()
def hello_world():
    game = Game()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DualResidualNetwork(num_channels=512, num_res_blocks=1).to(device)
    model.load_state_dict(torch.load("./models/recent/resnet_small.pth", map_location=torch.device("cpu")))
    mcts = MCTS(game=game, model=model, device=device, c_puct=1., dir_e=0)
    strategy = AlphaZeroStrategyV2(mcts)

    request_json = request.get_json(force=True)
    board, player = np.array(request_json["board"]), request_json["player"]
    state = game.get_canonical_form(board, player)
    action, policy = strategy.calculate_move(state, player)

    response = flask.jsonify({"action": action, "policy": policy.tolist()})
    return response
