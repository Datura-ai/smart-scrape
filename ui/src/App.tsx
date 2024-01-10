import React, { useEffect } from "react";
import { RecoilRoot } from 'recoil';
import { ToastContainer } from 'react-toastify';
import ChatPage from "./components/chat/ChatPage";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NotFound from "./components/NotFound";
import { TransitionGroup, CSSTransition } from 'react-transition-group';

function App() {
  useEffect(() => {
    if(localStorage.getItem('theme') === 'light'){
      document.documentElement.classList.remove("dark")
    }
  }, []);
  
  return (
    <div className="h-screen bg-regular">
      <RecoilRoot>
        <ToastContainer />
        <Router>
          <Routes>
            <Route path="/" element={
                <TransitionGroup>
                  <CSSTransition
                    in
                    timeout={500}
                    key={6}
                    classNames="page"
                    unmountOnExit>
                    <ChatPage />
                  </CSSTransition>
                </TransitionGroup>
            } />
            
            <Route path="*" element={
              <TransitionGroup>
                  <CSSTransition
                    in
                    timeout={500}
                    key={8}
                    classNames="page"
                    unmountOnExit>
                  <NotFound />
                </CSSTransition>
              </TransitionGroup>
            } />
          </Routes>
        </Router>
      </RecoilRoot>
    </div>
  );
}

export default App;