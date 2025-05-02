import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { auth, db, storage } from "./firebase"; // Ensure correct path
import { doc, getDoc, updateDoc, collection, getDocs } from "firebase/firestore";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { getDownloadURL, ref, uploadBytes, listAll } from "firebase/storage";

const CertificatesModal = ({ isOpen, onClose, certificates }) => {

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded-xl shadow-lg max-w-md w-full">
        <h2 className="text-lg font-semibold mb-4">Uploaded Certificates</h2>

        {certificates.length > 0 ? (
          <div className="space-y-2">
            {certificates.map((cert, index) => (
              <div key={index} className="p-2 border rounded-lg">
                <a
                  href={cert}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:underline"
                >
                  View Certificate {index + 1}
                </a>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">Nothing to see here.</p>
        )}

        <button
          onClick={onClose}
          className="mt-4 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Close
        </button>
      </div>
    </div>
  );
};

function Profilepage() {
  const [userData, setUserData] = useState(null);
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [certificates, setCertificates] = useState([]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        const userRef = doc(db, "users", user.uid);
        const userSnap = await getDoc(userRef);

        if (userSnap.exists()) {
          setUserData(userSnap.data());
        } else {
          navigate("/userinfo"); // Redirect if no data found
        }
      } else {
        navigate("/"); // Redirect to StartPage if not logged in
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      navigate("/");
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  const handlePhotoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !auth.currentUser) return; // ✅ Ensuring user exists before accessing UID

    try {
      const storageRef = ref(
        storage,
        `profilePictures/${auth.currentUser.uid}`
      );
      await uploadBytes(storageRef, file);
      const photoURL = await getDownloadURL(storageRef);

      // Update Firestore
      await updateDoc(doc(db, "users", auth.currentUser.uid), { photoURL });

      // Update UI
      setUserData((prev) => ({ ...prev, photoURL }));
    } catch (error) {
      console.error("Error uploading photo:", error);
    }
  };

  const handleViewCertificates = async () => {
    setIsModalOpen(true);

    if (!auth.currentUser) return;
    const userCertRef = ref(storage, `certificates/${auth.currentUser.uid}`);

    try {
      const fileList = await listAll(userCertRef);
      const urls = await Promise.all(
        fileList.items.map((item) => getDownloadURL(item))
      );
      setCertificates(urls);
    } catch (error) {
      console.error("Error fetching certificates:", error);
    }
  };

  useEffect(() => {
    const fetchAllRecommendedCourses = async () => {
      if (auth.currentUser) {
        const recommendationsRef = collection(db, "users", auth.currentUser.uid, "recommendations");
        const querySnapshot = await getDocs(recommendationsRef);

        // Combine all courses from the recommendations sub-collection
        const allCourses = [];
        querySnapshot.forEach((doc) => {
          const data = doc.data();
          if (data.courses) {
            allCourses.push(...data.courses);
          }
        });

        // Remove duplicates
        const uniqueCourses = [...new Set(allCourses)];

        // Update state with unique courses
        setUserData((prev) => ({
          ...prev,
          recommendedCourses: uniqueCourses,
        }));
      }
    };

    fetchAllRecommendedCourses();
  }, []);

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Profilepage.css";
    link.id = "profilepage-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("profilepage-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);

  return (
    <div id="webcrumbs">
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto bg-white rounded-3xl shadow-lg hover:shadow-xl transition-all duration-300">
          <div className="flex flex-col lg:flex-row">
            <div className="lg:w-1/3 relative">
              <div className="h-48 lg:h-full bg-gradient-to-r from-[#611BF8] to-[#611BF8]/80 rounded-t-3xl lg:rounded-l-3xl lg:rounded-tr-none">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 lg:left-1/2 lg:-translate-y-1/2">
                  <div className="relative w-40 h-40 rounded-full border-6 border-white bg-gray-200 shadow-lg overflow-hidden group">
                    <img
                      src={
                        userData?.photoURL ||
                        ""
                      }
                      alt="Profile"
                      className="w-full h-full object-cover transform hover:scale-110 transition-transform duration-300"
                    />

                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      id="photoUpload"
                      onChange={handlePhotoUpload}
                    />

                    <label
                      htmlFor="photoUpload"
                      className="absolute bottom-0 left-0 right-0 bg-black/50 py-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center gap-2 cursor-pointer"
                    >
                      <span className="material-symbols-outlined text-white">
                        edit
                      </span>
                      <span className="text-white text-sm">Edit Photo</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div className="lg:w-2/3 p-8 lg:p-12">
              <div className="text-center lg:text-left lg:ml-8">
                <h1 className="text-3xl lg:text-4xl font-bold mb-1">
                  {userData?.name || "User Name"}
                </h1>
                <p className="text-gray-500 mb-8">
                  ID: {auth.currentUser?.uid || "N/A"}
                </p>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 lg:ml-8">
                <div className="p-6 bg-gray-50 rounded-2xl hover:bg-gray-100 transition-colors duration-300 transform hover:scale-105">
                  <span className="material-symbols-outlined text-[#611BF8] mb-2">
                    calendar_today
                  </span>
                  <p className="text-sm text-gray-600">
                    Age: {userData?.age || "N/A"} years
                  </p>
                </div>
                <div className="p-6 bg-gray-50 rounded-2xl hover:bg-gray-100 transition-colors duration-300 transform hover:scale-105">
                  <span className="material-symbols-outlined text-[#611BF8] mb-2">
                    person
                  </span>
                  <p className="text-sm text-gray-600">
                    Gender: {userData?.gender || "N/A"}
                  </p>
                </div>
                <div className="p-6 bg-gray-50 rounded-2xl hover:bg-gray-100 transition-colors duration-300 transform hover:scale-105">
                  <span className="material-symbols-outlined text-[#611BF8] mb-2">
                    cake
                  </span>
                  <p className="text-sm text-gray-600">
                    DOB: {userData?.dob || "N/A"}
                  </p>
                </div>
                <div className="p-6 bg-gray-50 rounded-2xl hover:bg-gray-100 transition-colors duration-300 transform hover:scale-105">
                  <span className="material-symbols-outlined text-[#611BF8] mb-2">
                    school
                  </span>
                  <p className="text-sm text-gray-600">
                    10th: {userData?.boardPercentage || "N/A"}%
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-4 lg:ml-8">
                <div className="flex gap-4">
                  <button
                    onClick={() =>
                      navigate("/quiz", {
                        state: { userId: auth.currentUser?.uid },
                      })
                    }
                    className="flex-1 py-4 bg-[#611BF8] hover:bg-[#611BF8]/90 text-white rounded-2xl flex items-center justify-center gap-2 transform hover:scale-[1.02] transition-all duration-300 shadow-md hover:shadow-lg"
                  >
                    <span className="material-symbols-outlined">
                      assignment
                    </span>
                    Take Quiz
                  </button>

                  <button
                    onClick={handleViewCertificates}
                    className="flex-1 py-4 bg-[#611BF8] hover:bg-[#611BF8]/90 text-white rounded-2xl flex items-center justify-center gap-2 transform hover:scale-[1.02] transition-all duration-300 shadow-md hover:shadow-lg group"
                  >
                    <span className="material-symbols-outlined group-hover:rotate-12 transition-transform duration-300">
                      workspace_premium
                    </span>
                    <span className="text-center relative">
                      View Certificates
                      <span className="absolute -top-1 -right-2 w-2 h-2 bg-yellow-400 rounded-full animate-ping"></span>
                    </span>
                  </button>
                </div>

                <CertificatesModal
                  isOpen={isModalOpen}
                  onClose={() => setIsModalOpen(false)}
                  certificates={certificates}
                />

                <div className="mt-8">
                  <h2 className="text-2xl font-semibold mb-4">Recommended Courses</h2>
                  <div className="space-y-4">
                    {userData?.recommendedCourses?.length > 0 ? (
                      userData.recommendedCourses.map((course, index) => (
                        <p key={index} className="text-gray-600 text-center py-2">
                          {course}
                        </p>
                      ))
                    ) : (
                      <p className="text-gray-500 text-center py-8">Nothing to show here yet</p>
                    )}
                  </div>
                </div>

                <div className="mt-8">
                  <button
                    onClick={handleSignOut} // Calls the sign-out function
                    className="w-32 mx-auto py-2 bg-[#611BF8] hover:bg-[#611BF8]/90 text-white rounded-xl flex items-center justify-center gap-2 transform hover:scale-[1.02] transition-all duration-300 shadow-md hover:shadow-lg"
                  >
                    <span className="material-symbols-outlined">logout</span>
                    Sign Out
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Profilepage;
