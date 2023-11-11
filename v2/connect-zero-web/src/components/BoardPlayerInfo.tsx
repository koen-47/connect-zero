import React from 'react';

interface IProps {
    playerID: number
}

interface IState {
    
}


class BoardPlayerInfo extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.state = {
            
        }
    }

    render(): React.ReactNode {
        return (
            <div id={`player-info-container-${this.props.playerID}`} className="player-info-container">
                <div>
                    <div id={`color-${this.props.playerID}`} className="color">
                        {/* <p id={`player-id-text-${this.props.playerID}`}>Player 1</p>
                        <p id={`player-type-text-${this.props.playerID}`}>You</p> */}
                    </div>
                </div>
            </div>
        )
    }
}

export default BoardPlayerInfo;