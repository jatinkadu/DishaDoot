import React, { useState,useEffect } from 'react'
import { Link, useNavigate } from "react-router-dom";
import { auth, db } from "./firebase"; // Firebase import
import { signInWithEmailAndPassword } from "firebase/auth";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { toast } from 'react-toastify';
import './Login.css'; // Import CSS dynamically injected by vite-plugin-css-injected-by-js


function Login() {
  
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      const userRef = doc(db, "users", user.uid);
      const userSnap = await getDoc(userRef);

      if (!userSnap.exists() || !userSnap.data().isProfileComplete) {
        await setDoc(userRef, { isProfileComplete: false }, { merge: true });
        navigate("/userdetails");
      } else {
        navigate("/profilepage");
      }
    } catch (error) {
      toast.error(error.message);
    }
  };

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Login.css";
    link.id = "login-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("login-css");
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
            <h1 className="text-2xl font-bold mb-6 text-center">Login</h1>
            <div className="space-y-4">
              <div>
                <input
                  type="email"
                  placeholder="Email"
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Password"
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <span 
                  className="material-symbols-outlined absolute right-3 top-3.5 text-gray-400 cursor-pointer hover:text-gray-600"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? 'visibility' : 'visibility_off'}
                </span>
              </div>
              <div className="text-right">
                <a href="#" className="text-sm text-blue-500 hover:text-blue-600 hover:underline">Forgot password?</a>
              </div>
              <button 
                className="w-full bg-blue-500 text-white py-3 rounded-lg hover:bg-blue-600 transition duration-200"
                onClick={handleLogin}
              >
                Login
              </button>
              <div className="text-center text-sm text-gray-600">
                Don't have an account? 
                <Link to="/signup" className="text-blue-500 hover:text-blue-600 hover:underline ml-1">Signup</Link>
              </div>
              <div className="relative text-center my-4">
                <div className="absolute inset-y-1/2 w-full border-t border-gray-300"></div>
                <span className="relative bg-white px-2 text-sm text-gray-500">Or</span>
              </div>
              <button className="w-full bg-[#1877F2] text-white py-3 rounded-lg hover:bg-[#1864F2] transition duration-200 flex items-center justify-center gap-2">
                <i className="fa-brands fa-facebook text-xl"></i>
                Login with Facebook
              </button>
              <button className="w-full bg-white border border-gray-300 py-3 rounded-lg hover:bg-gray-50 transition duration-200 flex items-center justify-center gap-2">
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

export default Login
