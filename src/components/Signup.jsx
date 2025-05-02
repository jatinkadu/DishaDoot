import React, { useState,useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { auth } from './firebase'
import { createUserWithEmailAndPassword } from 'firebase/auth'
import { toast } from 'react-toastify';

function Signup() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const navigate = useNavigate()

  const handleSignup = async () => {
    if (password !== confirmPassword) {
      toast.error('Passwords do not match');
      return;
    }
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      navigate('/login');
    } catch (error) {
      toast.error(error.message);
    }
  };

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Signup.css";
    link.id = "signup-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("signup-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);

  return (
    <>
      <div id="webcrumbs"> 
        <div className="min-h-screen bg-blue-500 flex items-center justify-center">
          <div className="w-[400px] bg-white rounded-xl shadow-lg p-8">
            <h1 className="text-2xl font-bold text-center mb-6">Signup</h1>
            <div className="space-y-4">
              <input 
                type="email"
                placeholder="Email"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 outline-none" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <input
                type={showPassword ? 'text' : 'password'}
                placeholder="Create password"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 outline-none"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="Confirm password"  
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 outline-none"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
                <span 
                  className="material-symbols-outlined absolute right-3 top-3 text-gray-400 cursor-pointer hover:text-gray-600"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                >
                  {showConfirmPassword ? 'visibility' : 'visibility_off'}
                </span>
              </div>
              <button 
                className="w-full py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition duration-200"
                onClick={handleSignup}
              >
                Signup
              </button>
              <div className="text-center text-sm text-gray-600">
                Already have an account? 
                <Link to="/login" className="text-blue-500 hover:text-blue-600 ml-1">Login</Link>
              </div>
              <div className="relative text-center my-4">
                <hr className="border-gray-300" />
                <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white px-2 text-sm text-gray-500">Or</span>
              </div>
              <button className="w-full py-3 px-4 bg-[#1877F2] hover:bg-[#1864F2] text-white rounded-lg flex items-center justify-center gap-2 transition duration-200">
                <i className="fa-brands fa-facebook text-xl"></i>
                Login with Facebook
              </button>
              <button className="w-full py-3 px-4 bg-white hover:bg-gray-50 text-gray-600 rounded-lg border border-gray-300 flex items-center justify-center gap-2 transition duration-200">
                <img src="https://img.icons8.com/color/24/000000/google-logo.png" className="w-5 h-5" />
                Login with Google
              </button>
            </div>
          </div>
        </div> 
      </div>
    </>
  )
}

export default Signup
