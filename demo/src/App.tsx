import React from 'react';

import Main from './layouts/Main';
import GameContainer from './components/GameContainer';

function App() {
  return (
    <div className="App">
      <Main>
        <GameContainer />
      </Main>
    </div>
  );
}

export default App;
