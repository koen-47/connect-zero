import React from 'react';
type Props = {
    children?: React.ReactNode
}

const Main: React.FunctionComponent<Props> = ({ children } : Props) => {
    return (
        <div id="wrapper">
            <div id="main-container">
                <div id="main">
                    <header id="main-header">
                        <p><span>Connect</span> <span>Zero</span></p>
                    </header>
                    
                    <div id="main-content">
                        <div id="main-content-body">{children}</div>
                    </div>

                    <footer id="main-footer">

                    </footer>
                </div>
            </div>
            
        </div>
    )
};

export default Main;