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

#### Hyperparameters
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
        <td rowspan="5">MCTS</td>
        <td># simulations</td>
        <td colspan="3" align="center">100</td>
    </tr>
    <tr>
    </tr>
        <td>Temperature</td>
        <td>$$
\mathrm{CE}(p, y) = \begin{cases}
    -\log(p) & \text{if } y = 1 \\ % & is your "\tab"-like command (it's a tab alignment character)
    -\log(1-p) & \text{otherwise.}
\end{cases}
$$
</td>
        <td>0</td>
        <td>0</td>
    <tr>
        <td>$c_{puct}$</td>
        <td colspan="3" align="center">2</td>
    </tr>
    
</table>

### Results

#### Training

<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./experiment/plots/loss_accuracy_winrate_curves_dark.png">
    <img alt="" src="./experiment/plots/loss_accuracy_winrate_curves_light.png" />
</picture>
</div>

#### Experiments