import React, { useState, useRef, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { db, auth } from "./firebase";
import { doc, getDoc } from "firebase/firestore";
import { sendMessageToGemini } from "./ChatService";
import schemesData from "./schemes.json"; // Import Schemes JSON
import { ArrowLeft } from "lucide-react";

function Chatpage() {
  const location = useLocation();
  const selectedCourse = location.state?.selectedCourse || "Course Not Found";
  const selectedCourseDetails = location.state?.selectedCourseDetails || {};
  const selectedContent =
    location.state?.selectedContent || "No Content Available";

  const [input, setInput] = useState("");
  const [conversation, setConversation] = useState([
    { sender: "bot", text: "Hello! How can I assist you today?" },
  ]);
  const [recommendedSchemes, setRecommendedSchemes] = useState([]);
  const [recommendedCourses, setRecommendedCourses] = useState([]);
  const [loadingSchemes, setLoadingSchemes] = useState(true);
  const [userCaste, setUserCaste] = useState("General");
  const [userReligion, setUserReligion] = useState("Not Specified");

  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [conversation]);

  useEffect(() => {
    fetchUserData(auth.currentUser?.uid);
  }, []);

  useEffect(() => {
    if (userCaste && selectedCourse) {
      console.log("Fetching recommended schemes with:", {
        userCaste,
        userReligion,
        selectedCourse,
      });
      fetchRecommendedSchemes();
    }
  }, [userCaste, userReligion, selectedCourse]);

  useEffect(() => {
    if (selectedCourse) {
      console.log("Fetching recommended courses with:", { selectedCourse });
      fetchRecommendedCourses();
    }
  }, [selectedCourse]);

  // Fetch user data (caste & religion) from Firebase
  const fetchUserData = async (userId) => {
    if (!userId) return;
    try {
      const userRef = doc(db, "users", userId);
      const userSnap = await getDoc(userRef);

      if (userSnap.exists()) {
        const userInfo = userSnap.data();
        setUserCaste(userInfo.caste || "General");
        setUserReligion(userInfo.religion || "Not Specified");
      }
    } catch (error) {
      console.error("❌ Firestore Error:", error);
    }
  };

  // Fetch relevant schemes based on caste & course
  const fetchRecommendedSchemes = async () => {
    setLoadingSchemes(true);

    const casteSpecificSchemes = schemesData.filter((scheme) =>
      scheme["Scheme Eligibility"]
        ?.toLowerCase()
        .includes(userCaste.toLowerCase())
    );

    if (casteSpecificSchemes.length > 0) {
      setRecommendedSchemes(casteSpecificSchemes.slice(0, 3));
      setLoadingSchemes(false);
      return;
    }

    const generalSchemes = schemesData.filter(
      (scheme) =>
        !scheme["Scheme Eligibility"]?.toLowerCase().includes("sc") &&
        !scheme["Scheme Eligibility"]?.toLowerCase().includes("st") &&
        !scheme["Scheme Eligibility"]?.toLowerCase().includes("obc")
    );

    if (generalSchemes.length > 0) {
      setRecommendedSchemes(generalSchemes.slice(0, 3));
      setLoadingSchemes(false);
      return;
    }

    // If no local schemes, use Gemini
    const prompt = `The user belongs to the ${userCaste} caste and follows ${userReligion} religion.
Suggest government schemes they are eligible for based on:

Caste-based eligibility (if available).
General schemes (if no caste-specific scheme is found).
Relevance to their selected course: "${selectedCourse}".
User's Course Details:
Stream: ${selectedCourseDetails.Stream}
Degree Level: ${selectedCourseDetails["Degree Level"]}
Duration: ${selectedCourseDetails.Duration}
Eligibility: ${selectedCourseDetails.Eligibility}
Response Format:
🔹 Scheme Name: [Provide the scheme's official name]
🔹 Overview (Max 3 lines): [Briefly describe the scheme's purpose]
🔹 Benefits:

✅ [Key benefit 1]
✅ [Key benefit 2]
✅ [Key benefit 3]
🔹 Eligibility Criteria:
📌 [Who can apply?]
📌 [Caste/religion/income criteria, if any]
📌 [Educational qualifications]
🔹 Department: [Government body handling the scheme]:`;

    const response = await sendMessageToGemini(prompt);
    const parsedSchemes = response;

    setRecommendedSchemes(parsedSchemes.slice(0, 3));
    setLoadingSchemes(false);
  };

  // Fetch recommended courses (using Gemini or local dataset)
  const fetchRecommendedCourses = async () => {
    const prompt = `Based on the course "${selectedCourse}", recommend 3 similar courses a user can explore. Return only course names.`;
    const response = await sendMessageToGemini(prompt);
    setRecommendedCourses(response.split("\n").slice(0, 3)); // Extract top 3 courses
  };

  // Add debugging to log the conversation array after each update
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setConversation((prev) => {
      const updatedConversation = [...prev, userMessage];
      console.log("Updated Conversation:", updatedConversation);
      return updatedConversation;
    });
    setInput("");

    const botReply = await sendMessageToGemini(
      `Course: ${selectedCourse}\nContext: ${selectedContent}\nUser Question: ${input}`
    );
    const botMessage = { sender: "bot", text: botReply };

    setConversation((prev) => {
      const updatedConversation = [...prev, botMessage];
      console.log("Updated Conversation:", updatedConversation);
      return updatedConversation;
    });
  };

  // Updated colors for bot and user messages
  const renderMessage = (message, index) => {
    const isBot = message.sender === "bot";
    return (
      <div
        key={index}
        className={`flex ${isBot ? "justify-start ml-4 lg:ml-8" : "justify-end mr-4 lg:mr-8"} mb-4`}
      >
        <div
          className={`max-w-xs p-3 rounded-lg shadow-md text-black ${
            isBot ? "bg-indigo-50" : "bg-purple-50"
          }`}
        >
          {message.text}
        </div>
      </div>
    );
  };

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Chatpage.css";
    link.id = "chatpage-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("chatpage-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);

  return (
    <div id="webcrumbs">
      <div className="max-w-[1280px] mx-auto h-screen flex flex-col lg:flex-row bg-purple-50">
        {/* Left Side: Course Content + Details */}
        <section className="w-full lg:w-[70%] p-4 lg:p-6 border-b lg:border-b-0 lg:border-r border-gray-200">
          <div className="h-full rounded-xl bg-white shadow-sm p-4 lg:p-8 overflow-auto">
            <header className="mb-6 lg:mb-8">
              <h1 className="text-2xl lg:text-3xl font-bold mb-4">
                {selectedCourse}
              </h1>
              <p className="text-gray-700">
                <strong>Stream:</strong> {selectedCourseDetails.Stream}
              </p>
              <p className="text-gray-700">
                <strong>Degree Level:</strong>{" "}
                {selectedCourseDetails["Degree Level"]}
              </p>
              <p className="text-gray-700">
                <strong>Duration:</strong> {selectedCourseDetails.Duration}
              </p>
              <p className="text-gray-700">
                <strong>Eligibility:</strong>{" "}
                {selectedCourseDetails.Eligibility}
              </p>
            </header>

            {/* Recommended Courses Section */}
            <div className="mt-6">
              <h2 className="text-xl lg:text-2xl font-semibold mb-4">
                Recommended Schemes
              </h2>
              {loadingSchemes ? (
                <p className="text-gray-600 text-sm">🔄 Fetching schemes...</p>
              ) : recommendedSchemes.length > 0 ? (
                recommendedSchemes.map((scheme, index) => (
                  <details
                    key={index}
                    className="mb-4 bg-gray-100 rounded-lg shadow-md"
                  >
                    <summary className="p-3 font-semibold text-sm md:text-base">
                      {scheme["Scheme title"]}
                    </summary>
                    <div className="p-3 text-xs md:text-sm text-gray-700">
                      <p className="font-semibold">Overview:</p>
                      <ul className="list-disc pl-4">
                        {scheme["Scheme overview"]
                          .split(". ")
                          .slice(0, 3)
                          .map((point, i) => (
                            <li key={i}>{point.trim()}.</li>
                          ))}
                      </ul>

                      <p className="font-semibold mt-2">Department:</p>
                      <p>{scheme["scheme Department"]}</p>

                      <p className="font-semibold mt-2">Benefits:</p>
                      <ul className="list-disc pl-4">
                        {scheme["Scheme Benefits"]
                          .split(". ")
                          .slice(0, 3)
                          .map((point, i) => (
                            <li key={i}>{point.trim()}.</li>
                          ))}
                      </ul>

                      <p className="font-semibold mt-2">Eligibility:</p>
                      <ul className="list-disc pl-4">
                        {scheme["Scheme Eligibility"]
                          .split(". ")
                          .slice(0, 3)
                          .map((point, i) => (
                            <li key={i}>{point.trim()}.</li>
                          ))}
                      </ul>
                    </div>
                  </details>
                ))
              ) : (
                <p className="text-gray-600 text-sm">
                  No suitable schemes found.
                </p>
              )}
            </div>
            <button
              onClick={() => navigate(-1)}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg transition"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </button>
          </div>
        </section>

        {/* Right Side: Chatbot */}
        <section className="w-full lg:w-[30%] p-4 lg:p-6">
          <div className="h-full rounded-xl bg-white shadow-sm hover:shadow-md transition-shadow duration-300 p-4">
            <div className="flex items-center justify-between mb-4 lg:mb-6">
              <h2 className="text-lg lg:text-xl font-bold">Chat Support</h2>
              <span className="material-symbols-outlined cursor-pointer hover:scale-110 transition-transform text-indigo-600">
                settings
              </span>
            </div>
            <div
              className="h-[calc(100%-120px)] overflow-y-auto mb-4 space-y-4"
              ref={chatContainerRef}
              style={{ maxHeight: '400px', overflowY: 'scroll' }}
            >
              {conversation.map((msg, index) => renderMessage(msg, index))}
            </div>
            <div className="flex items-center space-x-2">
              <textarea
                ref={inputRef}
                className="w-full px-3 lg:px-4 py-2 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all duration-300 resize-none h-10"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
              <button
                onClick={handleSendMessage}
                className="p-2 rounded-lg bg-indigo-500 hover:bg-indigo-600 transition-colors duration-300"
              >
                <span className="material-symbols-outlined text-white">send</span>
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Chatpage;
