import React from "react";
import { Navigate } from "react-router-dom";
import { auth } from "./firebase"; // Ensure correct path to Firebase
import { onAuthStateChanged } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import { db } from "./firebase"; // Ensure correct path to Firebase

const ProtectedRoute = ({ children, requiredProgress }) => {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [userProgress, setUserProgress] = React.useState(null);
  const [isProfileComplete, setIsProfileComplete] = React.useState(false);

  React.useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        setIsAuthenticated(true);
        const userDoc = await getDoc(doc(db, "users", user.uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          setUserProgress(userData.progress || "none");
          setIsProfileComplete(userData.isProfileComplete || false);
        }
      } else {
        setIsAuthenticated(false);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  if (loading) {
    return (<div>Loading...</div>); // Show a loading indicator while checking auth state
  }

  if (!isAuthenticated) {
    return <Navigate to="/" />; // Redirect to login if not authenticated
  }

  if (!isProfileComplete) {
    return <Navigate to="/userdetails" />; // Redirect to user details if profile is incomplete
  }

  if (requiredProgress && userProgress !== requiredProgress) {
    return <Navigate to="/profilepage" />; // Redirect to profile if progress does not match
  }

  return children;
};

export default ProtectedRoute;
