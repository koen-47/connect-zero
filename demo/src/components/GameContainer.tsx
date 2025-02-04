import React from 'react';

import Board from './Board';

interface IProps {

}

interface IState {

}


class GameContainer extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.state = {
            playerTurn: 1
        }
    }

    render(): React.ReactNode {
        return (
            <div id="game-container">
                <Board />
            </div>
        )
    }
}

export default GameContainer;