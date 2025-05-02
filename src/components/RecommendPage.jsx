import React, { useState, useEffect } from "react";
import { db, auth } from "./firebase";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import courseDetailsData from "./courses.json"; // Import JSON file


const RecommendPage = () => {
  const [courses, setCourses] = useState([]);
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();
  const authInstance = getAuth();

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/RecommendPage.css";
    link.id = "recommendpage-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("recommendpage-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(authInstance, async (user) => {
      if (user) {
        fetchUserData(user.uid);
      } else {
        setErrorMessage("❌ Please log in to view recommendations.");
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  // Ensure back button navigates to profile page
  useEffect(() => {
    const handlePopState = (event) => {
      navigate("/profilepage");
    };

    window.addEventListener("popstate", handlePopState);

    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, [navigate]);

  const fetchUserData = async (userId) => {
    try {
      const userRef = doc(db, "users", userId);
      const userSnap = await getDoc(userRef);

      if (!userSnap.exists()) {
        setErrorMessage("❌ User data not found.");
        setLoading(false);
        return;
      }

      const userInfo = userSnap.data();
      const hobbies = userInfo.hobbies || [];
      const quizScore = userInfo.score || 0;
      setUserData({ hobbies, quizScore });

      fetchCourseRecommendations(userId, hobbies, quizScore);
    } catch (error) {
      setErrorMessage(`❌ Firestore Error: ${error.message}`);
      setLoading(false);
    }
  };

  const fetchCourseRecommendations = async (userId, hobbies, score) => {
    try {
      const response = await fetch("http://localhost:5001/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hobbies, score }),
      });

      const data = await response.json();
      if (data.recommendations) {
        setCourses(data.recommendations);
        storeRecommendations(userId, data.recommendations);
      } else {
        setErrorMessage("❌ No recommendations received.");
      }
    } catch (error) {
      setErrorMessage(`❌ Flask API Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const storeRecommendations = async (userId, recommendations) => {
    try {
      const recommendationsRef = doc(
        db,
        "users",
        userId,
        "recommendations",
        "latest"
      );
      await setDoc(recommendationsRef, {
        courses: recommendations,
        timestamp: new Date(),
      });
      console.log("✅ Recommendations stored successfully for user:", userId);
    } catch (error) {
      console.error("❌ Error storing recommendations:", error);
    }
  };

  const handleNavigate = (courseName) => {
    const courseDetails = getCourseDetails(courseName);
    navigate("/chatpage", {
      state: {
        selectedCourse: courseName,
        selectedCourseDetails: courseDetails, // Send full details
      },
    });
  };
  

  const getCourseDetails = (courseName) => {
    const course = courseDetailsData.find(
      (c) => c["Program Name"] === courseName
    );
    return course || {
      Stream: "N/A",
      "Degree Level": "N/A",
      Duration: "N/A",
      Eligibility: "N/A",
    };
  };

  return (
    <div id="webcrumbs">
      <div className="flex justify-center items-center min-h-screen">
        <div className="w-full md:w-[600px] p-4 md:p-6 bg-neutral-50 rounded-lg">
          {loading ? (
            <p className="text-center text-lg font-semibold">
              Loading course recommendations...
            </p>
          ) : errorMessage ? (
            <p className="text-center text-lg font-semibold text-red-500">
              {errorMessage}
            </p>
          ) : courses.length > 0 ? (
            courses.map((course, index) => {
              const details = getCourseDetails(course);
              return (
                <details
                  key={index}
                  className="mb-4 bg-white rounded-lg shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group hover:bg-indigo-50"
                >
                  <summary className="p-3 md:p-4 font-semibold rounded-lg hover:bg-indigo-100 text-black text-sm md:text-base flex items-center">
                    <span className="flex-1 truncate">
                      {index + 1}. {course}
                    </span>
                    <span className="material-symbols-outlined group-hover:text-indigo-600">
                      expand_more
                    </span>
                  </summary>
                  <div className="p-3 md:p-4 pt-2 text-black text-sm md:text-base">
                    <p>
                      <strong>Course Name:</strong> {details["Program Name"]}
                    </p>
                    <p>
                      <strong>Stream:</strong> {details.Stream}
                    </p>
                    <p>
                      <strong>Degree Level:</strong> {details["Degree Level"]}
                    </p>
                    <p>
                      <strong>Duration:</strong> {details.Duration}
                    </p>
                    <p>
                      <strong>Eligibility:</strong> {details.Eligibility}
                    </p>
                    <div className="flex justify-end mt-4">
                      <span
                        className="material-symbols-outlined hover:scale-110 transition-transform cursor-pointer hover:text-indigo-600"
                        onClick={() => handleNavigate(course)}
                      >
                        arrow_forward
                      </span>
                    </div>
                  </div>
                </details>
              );
            })
          ) : (
            <p className="text-center text-lg font-semibold">
              No recommendations found.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecommendPage;
