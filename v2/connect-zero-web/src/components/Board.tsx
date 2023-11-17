import { access } from 'fs';
import React from 'react';

import BoardPlayerInfo from './BoardPlayerInfo';
import checkFourAdjacent from "../utils/GameUtils"

interface IProps {

}

interface IState {
    board: Array<Array<number>>,
    currentColumnHover: number,
    currentPlayer: number,
    humanID: number,
    isGameOver: boolean,
    isRequesting: boolean
}


class Board extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.state = {
            board: [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0]],
            currentColumnHover: -1,
            currentPlayer: 1,
            humanID: 1,
            isGameOver: false,
            isRequesting: false
        }
    }

    componentDidMount(): void {
        if (this.state.humanID == -1) {
            console.log("Requesting move from AI...")
            this.fetchConnectZeroMove().then((action) => {
                this.playMove(action)
            })
        }
    }

    getCellCornerClass(i: number, j: number): string {
        if (i == 0 && j == 0) {
            return "board-cell-top-left"
        } else if (i == 0 && j == this.state.board[0].length-1) {
            return "board-cell-top-right"
        } else if (i == this.state.board.length-1 && j == 0) {
            return "board-cell-bottom-left"
        } else if (i == this.state.board.length-1 && j == this.state.board[0].length-1) {
            return "board-cell-bottom-right"
        }
        return ""
    }

    getCellValueClass(i: number, j: number): string {
        const cellValue = this.state.board[i][j]
        if (cellValue == 1) {
            return "board-cell-p1"
        } else if (cellValue == -1) {
            return "board-cell-p2"
        }
        return "board-cell-no-player"
    }

    getColumnHighlight(col: number): string {
        if (this.state.currentColumnHover == col) {
            if (this.state.currentPlayer == 1) {
                return "board-cell-highlight-p1"
            }
            return "board-cell-highlight-p2"
        }
        return ""
    }

    playMove(col: number) {
        for (var i = this.state.board.length-1; i >= 0; i--) {
            var newBoard = this.state.board
            const cellValue = newBoard[i][col]
            if (cellValue == 0) {
                newBoard[i][col] = this.state.currentPlayer
                this.setState({board: newBoard, currentPlayer: -this.state.currentPlayer})
                break
            }
        }

        const isGameOver = checkFourAdjacent(this.state.board)
        if (isGameOver) {
            alert(`Game over! (${isGameOver})`)
        }

        return isGameOver;
    }

    fetchConnectZeroMove(): Promise<any> {
        return fetch("https://koen-kraaijveld.onrender.com/connect_zero", {
            method: "POST",
            body: JSON.stringify({
                "board": this.state.board,
                "player": 1
            }),
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            }
        })
        .then(response => response.json()
            .then(data => {
                return data.action
            })
        )
    }

    playTurn(col: number) {
        var isGameOver = checkFourAdjacent(this.state.board)
        if (!isGameOver) {
            isGameOver = this.playMove(col)
            this.setState({isGameOver: isGameOver})        
        }

        if (!isGameOver) {
            console.log(`Sending board...`)
            console.log(this.state.board)
            this.setState({isRequesting: true})
            this.fetchConnectZeroMove().then((action) => {
                isGameOver = this.playMove(action)
                this.setState({isRequesting: false, isGameOver: isGameOver})
            })
        }
    }

    render(): React.ReactNode {
        return (
            <div id="board-container">
                <table id="board" className={this.state.isRequesting || this.state.isGameOver ? "disabled" : ""}>
                    {[...Array(this.state.board.length)].map((_, i) => (
                        <tr>
                            {[...Array(this.state.board[0].length)].map((_, j) => (
                                <td className={`${this.getCellCornerClass(i, j)} ${this.getColumnHighlight(j)}`} 
                                onMouseEnter={() => this.setState({currentColumnHover: j})}
                                onMouseLeave={() => this.setState({currentColumnHover: -1})}
                                onClick={() => this.playTurn(j)}>
                                    <div className={`board-cell ${this.getCellValueClass(i, j)}`} />
                                </td>
                            ))}
                        </tr>
                    ))}
                    
                </table>
                <div id="board-info-container">
                    <BoardPlayerInfo playerID={1} isPlaying={this.state.currentPlayer == 1} humanID={this.state.humanID}/>
                    <BoardPlayerInfo playerID={2} isPlaying={this.state.currentPlayer == -1} humanID={this.state.currentPlayer}/>
                </div>
            </div>
        )
    }
}

export default Board;
