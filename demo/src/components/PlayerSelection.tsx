import React from 'react';

interface IProps {

}

interface IState {

}


class PlayerSelection extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.state = {
            playerTurn: 1
        }
    }
}

export default PlayerSelection;
