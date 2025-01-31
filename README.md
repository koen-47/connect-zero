<div align="center">
<h1>ConnectZero</h1>
  
An adaptation to DeepMind’s AlphaZero model for Connect Four through a
combination of deep reinforcement learning and tree search algorithms
\
\
[**🎮 Demo** (works best on Chrome)](https://connect-zero.onrender.com/) 

</div>


## Methodology

The following section offers a concise overview of the methodology used 
in the development of ConnectZero. 
It includes links to the relevant code and results for reference. 
While it highlights the key aspects of the approach, 
it is not intended to be exhaustive (please refer to the 
code for a complete and detailed overview).

[WORK IN PROGRESS]

### Model Development

#### Input Features

The input consists of a 3 $\times$ 6 $\times$ 7 image. The last two dimensions refer to the size of a Connect Four board (6 $\times$ 7).
The first input plane corresponds to all pieces belonging to player 1 (1 if there is a piece, 0 otherwise). 
The second plane is the same, but for player 2.
The last plane shows which player is about to play (1 if player 1, 0 if player 2).

#### Hyperparameters

The following table shows the hyperparameter setup used, all of which were 
tuned manually.

<div align="center">
<table>
    <tr>
        <th rowspan="2">Category</th>
        <th rowspan="2">Hyperparameter</th>
        <th colspan="3">Phase</th>
    </tr>
    <tr>
        <th>Self-play</th>
        <th>Evaluation</th>
        <th>Experimentation</th>
    </tr>
    <tr>
        <td rowspan="4">Training loop</td>
        <td># iterations</td>
        <td colspan="3">80</td>
    </tr>
    <tr>
        <td># self-play episodes</td>
        <td>250</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td># evaluation games</td>
        <td>-</td>
        <td colspan="2">50</td>
    </tr>
    <tr>
        <td>Model acceptance rate</td>
        <td>-</td>
        <td>>0.55</td>
        <td>-</td>
    </tr>
    <tr>
        <td rowspan="5">MCTS</td>
        <td># simulations</td>
        <td colspan="3">100</td>
    </tr>
    <tr>
        <td>Temperature</td>
        <td>1 (if # turn < 15) <br> 0 otherwise</td>
        <td colspan="2">0</td>
    </tr>
    <tr>
        <td>$c_{puct}$</td>
        <td colspan="3">2</td>
    </tr>
    <tr>
        <td>Dirichlet $\alpha$</td>
        <td>0.5</td>
        <td colspan="2">0</td>
    </tr>
    <tr>
        <td>Dirichlet $\epsilon$</td>
        <td>0.25</td>
        <td colspan="2">0</td>
    </tr>
    <tr>
        <td rowspan="4">Network</td>
        <td>Type</td>
        <td colspan="3">ResNet</td>
    </tr>
    <tr>
        <td># blocks</td>
        <td colspan="3">5</td>
    </tr>
    <tr>
        <td># filters</td>
        <td colspan="3">128</td>
    </tr>
    <tr>
        <td>Dropout probability</td>
        <td>0.3</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td rowspan="3">Learning</td>
        <td># epochs</td>
        <td>10</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Learning rate</td>
        <td>10<sup>-3</sup></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Weight decay</td>
        <td>10<sup>-3</sup></td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>
</div>

### Results

#### Training

<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./experiment/plots/loss_accuracy_winrate_curves_dark.png">
    <img alt="" src="./experiment/plots/loss_accuracy_winrate_curves_light.png" />
</picture>
</div>

#### Experiments