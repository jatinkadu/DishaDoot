import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Startpage from './components/Startpage';
import Login from "./components/Login";
import Signup from "./components/Signup";
import Userdetails from './components/Userdetails';
import Profilepage from './components/Profilepage'
import Quizpage from './components/Quizpage';
import Chatpage from './components/Chatpage';
import RecommendPage from './components/RecommendPage';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import ProtectedRoute from './components/ProtectedRoute'// Import the ProtectedRoute component

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Startpage />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/userdetails" element={<Userdetails />} />
        <Route path="/profilepage" element={<ProtectedRoute><Profilepage /></ProtectedRoute>} />
        <Route path="/quiz" element={<Quizpage />} />
        <Route path="/recommend" element={<ProtectedRoute requiredProgress="quizCompleted"><RecommendPage /></ProtectedRoute>} />
        <Route path="/chatpage" element={<ProtectedRoute requiredProgress="quizCompleted"><Chatpage /></ProtectedRoute>} />

      </Routes>
      <ToastContainer position="top-center" />
    </Router>    
    
    
  );
}

export default App;
