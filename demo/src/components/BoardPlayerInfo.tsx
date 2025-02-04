import React from 'react';

import { ImSpinner2 } from "react-icons/im"

interface IProps {
    playerID: number,
    isPlaying: boolean,
    humanID: number
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
                <div id={`color-${this.props.playerID}`} className={`color ${this.props.isPlaying && 'is-playing'}`}>
                    <span className={`player-title-${this.props.playerID}`}>
                        {`Player ${this.props.playerID}`}<br/>
                        <span>{this.props.playerID == this.props.humanID ? `You` : `AI`}</span>
                    </span>
                    {this.props.isPlaying && <ImSpinner2 className="spinner" size={40} />}
                </div>
            </div>
        )
    }
}

export default BoardPlayerInfo;