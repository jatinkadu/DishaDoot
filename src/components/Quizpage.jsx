import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import screenfull from "screenfull";
import { db } from "./firebase";
import { collection, addDoc, doc, updateDoc, arrayUnion, serverTimestamp } from "firebase/firestore";
import { toast } from "react-toastify";

const Quizpage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const userId = location.state?.userId || null; // Get userId from ProfilePage
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [timer, setTimer] = useState(1800); // 30 minutes in seconds
  const [isStarted, setIsStarted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const [showPopup, setShowPopup] = useState(false);
  const [scoreMessage, setScoreMessage] = useState("");

  useEffect(() => {
    console.log("Received User ID in Quiz Page:", userId);
  }, [userId]);

  // Fetch Questions from Flask API
  useEffect(() => {
    fetch("http://localhost:5000/api/get-questions")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setQuestions(data);
          setLoading(false);
        } else {
          console.error("❌ Invalid API response:", data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error("❌ Fetch Error:", err);
        setLoading(false);
      });
  }, []);

  // Ensure Full-Screen Mode & Timer
  useEffect(() => {
    let interval;
    if (isStarted) {
      if (screenfull.isEnabled) {
        screenfull.request();
      }
      interval = setInterval(() => {
        setTimer((prev) => (prev > 0 ? prev - 1 : 0));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isStarted]);

  // Ensure full-screen mode is exited when navigating away
  useEffect(() => {
    return () => {
      if (screenfull.isEnabled && screenfull.isFullscreen) {
        screenfull.exit();
      }
    };
  }, []);

  const handleAnswerChange = (qIndex, answer) => {
    setAnswers((prevAnswers) => ({
      ...prevAnswers,
      [qIndex]: answer,
    }));
  };

  const handleSubmit = async () => {
    if (!userId) {
      setErrorMessage("❌ Please log in before submitting the quiz.");
      return;
    }

    const unansweredQuestions = questions.filter(
      (q) => !answers.hasOwnProperty(q.question)
    );

    if (unansweredQuestions.length > 0) {
      const firstUnanswered = unansweredQuestions[0];
      const questionIndex = questions.findIndex(
        (q) => q.question === firstUnanswered.question
      );

      setErrorMessage(
        `❌ Please answer all questions. Redirecting to question ${questionIndex + 1}.`
      );

      const questionElement = document.getElementById(
        `question-${firstUnanswered.question}`
      );
      if (questionElement) {
        questionElement.scrollIntoView({ behavior: "smooth" });
      }
      return;
    }

    const correctAnswers = questions.filter(
      (q) => q.correct_answer === answers[q.question]
    );
    const scorePercentage = (
      (correctAnswers.length / questions.length) * 100
    ).toFixed(2);

    const recommendedCourses = ["Course 1", "Course 2", "Course 3"];

    try {
      const recommendationsRef = collection(db, "users", userId, "recommendations");
      await addDoc(recommendationsRef, {
        courses: recommendedCourses,
        timestamp: serverTimestamp(),
      });

      const userRef = doc(db, "users", userId);
      await updateDoc(userRef, { progress: "quizCompleted" });

      toast.success(`✅ Quiz Submitted Successfully! Your Score: ${scorePercentage}%`, {
        position: "top-center",
        autoClose: 3000,
      });

      navigate("/recommend"); // Ensure navigation happens after progress update
    } catch (error) {
      console.error("❌ Firestore Error:", error);
      setErrorMessage("❌ Error saving the quiz. Try again.");
    }
  };

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Quizpage.css";
    link.id = "quizpage-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("quizpage-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);

  return (
    <div id="webcrumbs" className="relative flex flex-col items-center min-h-screen w-full bg-white">
      {/* Timer is fixed at the right and always visible when scrolling */}
      {isStarted && (
        <div
          className="fixed top-4 right-4 z-50"
          style={{ position: 'fixed', top: '16px', right: '16px', zIndex: 50 }} // Ensure proper positioning
        >
          <div className="bg-blue-500 text-white px-6 py-3 rounded-lg shadow-md flex items-center space-x-2">
            <span className="material-symbols-outlined">timer</span>
            <span className="font-semibold">
              Time left: {Math.floor(timer / 60)}:{String(timer % 60).padStart(2, "0")}
            </span>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-lg p-8 relative w-full max-w-5xl">
        {!isStarted ? (
          <div className="text-center">
            <h2 className="text-2xl font-bold">Quiz Instructions</h2>
            <p className="mt-2">Complete all questions within 30 minutes.</p>
            <button
              className="mt-4 px-8 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600" // Changed `rounded-full` to `rounded-lg` for slightly rounded corners
              onClick={() => setIsStarted(true)}
              style={{ display: 'block', margin: '0 auto', visibility: 'visible' }}
            >
              Start Quiz
            </button>
          </div>
        ) : loading ? (
          <p className="text-center text-lg font-semibold">Loading questions...</p>
        ) : questions.length === 0 ? (
          <p className="text-center text-lg font-semibold text-red-500">
            No questions available. Please check the backend.
          </p>
        ) : (
          <form className="space-y-8 text-left mt-6">
            {[...new Set(questions.map((q) => q.section))].map((section, index) => (
              <div key={index}>
                <h2 className="text-2xl font-bold mt-6">{section} Section</h2>
                {questions
                  .filter((q) => q.section === section)
                  .map((q, i) => (
                    <div
                      key={i}
                      id={`question-${q.question}`} // Added unique id for each question
                      className="space-y-6 p-6 bg-gray-50 rounded-lg"
                    >
                      {/* Numbered Questions & Mandatory Asterisk */}
                      <p className="text-lg font-semibold mb-4 flex justify-between items-center">
                        {i + 1}. {q.question} <span className="text-red-500">*</span>
                      </p>
                      <div className="space-y-3">
                        {q.options.map((option, idx) => (
                          <label
                            key={idx}
                            className={`flex items-center space-x-3 p-4 border rounded-lg transition-colors cursor-pointer ${
                              answers[q.question] === option ? "bg-blue-100" : "hover:bg-gray-100"
                            }`}
                          >
                            <input
                              type="radio"
                              name={`question-${q.question}`}
                              className="w-4 h-4 accent-blue-500"
                              value={option}
                              checked={answers[q.question] === option}
                              onChange={() => handleAnswerChange(q.question, option)}
                            />
                            <span>{option}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            ))}
            {errorMessage && (
              <div className="text-red-500 font-semibold text-center">{errorMessage}</div>
            )}
            <div className="flex justify-center mt-8">
              {/* Adjusted padding for the Submit Quiz button to prevent overflow */}
              <button
                type="button"
                className="bg-blue-500 text-white px-8 py-3 rounded-lg font-semibold transform hover:scale-[1.02] hover:bg-blue-600 transition-all shadow-md hover:shadow-lg" // Reduced padding slightly
                onClick={handleSubmit}
                disabled={Object.keys(answers).length !== questions.length}
                style={{ display: 'block', margin: '0 auto', visibility: 'visible' }}
              >
                Submit Quiz
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default Quizpage;
