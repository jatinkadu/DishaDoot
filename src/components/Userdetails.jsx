import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { auth, db } from "./firebase"; // Firebase import
import { doc, getDoc, setDoc } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";
import {
  getStorage,
  ref,
  uploadBytesResumable,
  getDownloadURL,
} from "firebase/storage"; // Firebase Storage
import { toast } from "react-toastify";

function Userdetails() {
  const [userData, setUserData] = useState({
    name: "",
    age: "",
    city: "",
    gender: "",
    dob: "",
    address: "",
    state: "",
    boardType: "",
    maths: "",
    science: "",
    boardPercentage: "",
    hobbies: [], // Make sure hobbies is an array
    achievements: "",
    religion: "",
    caste: "",
  });

  const [loading, setLoading] = useState(true);
  const [dropdownOpen, setDropdownOpen] = useState(false); // State for dropdown
  const navigate = useNavigate();

  // Fetch user data and populate the form
  useEffect(() => {
    setLoading(true);

    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (currentUser) {
        const fetchUserData = async () => {
          try {
            const userRef = doc(db, "users", currentUser.uid);
            const userSnap = await getDoc(userRef);

            if (userSnap.exists()) {
              setUserData({
                name: userSnap.data().name || "",
                age: userSnap.data().age ? userSnap.data().age.toString() : "",
                city: userSnap.data().city || "",
                gender: userSnap.data().gender || "",
                dob: userSnap.data().dob || "",
                address: userSnap.data().address || "",
                state: userSnap.data().state || "",
                boardType: userSnap.data().boardType || "",
                maths: userSnap.data().maths ? userSnap.data().maths.toString() : "",
                science: userSnap.data().science ? userSnap.data().science.toString() : "",
                boardPercentage: userSnap.data().boardPercentage
                  ? userSnap.data().boardPercentage.toString()
                  : "",
                hobbies: userSnap.data().hobbies || [], // Ensuring it's always an array
                achievements: userSnap.data().achievements || "",
                religion: userSnap.data().religion || "", // Not compulsory, default to empty string
                caste: userSnap.data().caste || "", // Not compulsory, default to empty string
                certificates: userSnap.data().certificates || [], // Ensuring it's always an array
              });

              if (userSnap.data().isProfileComplete) {
                navigate("/blankpage");
              }
            }
          } catch (error) {
            console.error("Error fetching user data:", error);
          } finally {
            setLoading(false);
          }
        };

        fetchUserData();
      } else {
        setLoading(false);
      }
    });

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [navigate]);

  // Handle changes in input fields
  const handleChange = (e) => {
    setUserData({ ...userData, [e.target.name]: e.target.value || "" });
  };

  // Handle hobby selection
  const handleHobbySelect = (hobby) => {
    if (userData.hobbies.length < 5 && !userData.hobbies.includes(hobby)) {
      setUserData({ ...userData, hobbies: [...userData.hobbies, hobby] });
    }
    setDropdownOpen(false); // Close the dropdown after selecting an option
  };

  // Handle hobby removal
  const handleHobbyRemove = (hobby) => {
    setUserData({
      ...userData,
      hobbies: userData.hobbies.filter((h) => h !== hobby), // Remove the selected hobby
    });
  };

  // Save data to Firebase
  const handleSave = async () => {
    const user = auth.currentUser;
    if (user) {
      await setDoc(
        doc(db, "users", user.uid),
        { ...userData, isProfileComplete: true },
        { merge: true }
      );
      navigate("/profilepage");
    }
  };

  // Handle file upload for certificates
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const storage = getStorage();
      const storageRef = ref(storage, `certificates/${auth.currentUser.uid}/${file.name}`);

      const uploadTask = uploadBytesResumable(storageRef, file);

      uploadTask.on(
        "state_changed",
        (snapshot) => {
          // Optional: Handle progress
        },
        (error) => {
          console.error("Error uploading file:", error);
        },
        () => {
          getDownloadURL(uploadTask.snapshot.ref).then((downloadURL) => {
            setUserData({
              ...userData,
              certificates: [...(userData.certificates || []), downloadURL], // Ensure it's always an array
            });
          });
        }
      );
    }
  };

  // Validate and save form
  const handleValidationAndSave = () => {
    const requiredFields = [
      "name",
      "age",
      "city",
      "gender",
      "dob",
      "state",
      "boardType",
      "maths",
      "science",
      "boardPercentage",
      "hobbies",
    ];

    let firstErrorField = null;

    for (const field of requiredFields) {
      if (!userData[field] || (field === "hobbies" && userData.hobbies.length === 0)) {
        firstErrorField = field;
        toast.error(`Please fill out the required field: ${firstErrorField}`, {
          position: "top-center",
          autoClose: 3000,
        });
        document.querySelector(`[name="${firstErrorField}"]`)?.scrollIntoView({ behavior: "smooth" });
        return;
      }
    }

    // Age validation
    const age = parseInt(userData.age);
    if (isNaN(age) || age < 1 || age > 50) {
      toast.error("Age must be between 1 and 50.", {
        position: "top-center",
        autoClose: 3000,
      });
      document.querySelector(`[name="age"]`)?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    // DOB and Age matching validation
    const dob = new Date(userData.dob);
    const today = new Date();
    let calculatedAge = today.getFullYear() - dob.getFullYear();
    const monthDifference = today.getMonth() - dob.getMonth();
    const dayDifference = today.getDate() - dob.getDate();

    if (monthDifference < 0 || (monthDifference === 0 && dayDifference < 0)) {
      calculatedAge -= 1;
    }

    if (calculatedAge !== parseInt(userData.age)) {
      toast.error("Age and Date of Birth do not match.", {
        position: "top-center",
        autoClose: 3000,
      });
      document.querySelector(`[name="dob"]`)?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    // Maths validation
    const maths = parseInt(userData.maths);
    if (isNaN(maths) || maths < 0 || maths > 100) {
      toast.error("Maths marks must be between 0 and 100.", {
        position: "top-center",
        autoClose: 3000,
      });
      document.querySelector(`[name="maths"]`)?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    // Science validation
    const science = parseInt(userData.science);
    if (isNaN(science) || science < 0 || science > 100) {
      toast.error("Science marks must be between 0 and 100.", {
        position: "top-center",
        autoClose: 3000,
      });
      document.querySelector(`[name="science"]`)?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    // Board percentage validation
    const boardPercentage = parseFloat(userData.boardPercentage);
    if (isNaN(boardPercentage) || boardPercentage < 0 || boardPercentage > 100) {
      toast.error("Board percentage must be between 0 and 100.", {
        position: "top-center",
        autoClose: 3000,
      });
      document.querySelector(`[name="boardPercentage"]`)?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    // If all validations pass
    toast.success("Details saved successfully!", {
      position: "top-center",
      autoClose: 3000,
    });
    handleSave();
  };

  const citiesList = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Jaipur",
    "Lucknow",
    "Ahmedabad",
    "Surat",
    "Bhopal",
    "Indore",
    "Chandigarh",
    "Guwahati",
    "Patna",
    "Nagpur",
    "Dehradun",
    "Shimla",
  ];

  const statesList = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Lakshadweep",
    "Delhi",
    "Puducherry",
    "Ladakh",
    "Jammu and Kashmir",
  ];

  const religionsList = ["Hindu", "Muslim", "Christian", "Sikh", "Other"];

  const castesList = ["General", "OBC", "SC", "ST", "Other"];

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Userdetails.css";
    link.id = "userdetails-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("userdetails-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);

  return (
    <>
      <div id="webcrumbs">
        <div className="flex flex-col md:flex-row min-h-screen bg-gray-50">
          <div className="w-full md:w-[280px] bg-white shadow-sm p-4 md:p-6 md:sticky top-0 md:h-screen">
            {/* Navigation */}
            <nav className="space-y-2">
              <a
                href="#personal"
                className="block w-full text-left px-4 py-2 rounded bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors"
              >
                Personal Details
              </a>
              <a
                href="#educational"
                className="block w-full text-left px-4 py-2 rounded hover:bg-gray-100 transition-colors"
              >
                Educational Details
              </a>
              <a
                href="#interests"
                className="block w-full text-left px-4 py-2 rounded hover:bg-gray-100 transition-colors"
              >
                Personal Interests and Hobbies
              </a>
              <a
                href="#religion"
                className="block w-full text-left px-4 py-2 rounded hover:bg-gray-100 transition-colors"
              >
                Religion and Caste
              </a>
            </nav>
          </div>

          <div className="flex-1 p-4">
            <div className="max-w-[900px] mx-auto space-y-8">
              {/* Personal Info */}
              <section
                id="personal"
                className="bg-white rounded-lg shadow-sm p-4 md:p-8"
              >
                <div className="flex flex-col md:flex-row items-start justify-between">
                  <div className="flex-1 w-full">
                    <h2 className="text-2xl font-semibold mb-6">
                      Personal Details
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-4">
                      {/* Name */}
                      <div className="col-span-full">
                        <label className="block mb-1">
                          Name: <span className="text-red-500">*</span>
                        </label>
                        <input
                          required
                          type="text"
                          placeholder="Enter your name"
                          name="name"
                          value={userData.name}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        />
                      </div>

                      {/* Age */}
                      <div className="col-span-1">
                        <label className="block mb-1">
                          Age: <span className="text-red-500">*</span>
                        </label>
                        <input
                          required
                          type="number"
                          placeholder="Enter your age"
                          name="age"
                          value={userData.age}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        />
                      </div>

                      {/* City Dropdown */}
                      <div className="col-span-1 md:col-span-2">
                        <label className="block mb-1">
                          City: <span className="text-red-500">*</span>
                        </label>
                        <select
                          required
                          name="city"
                          value={userData.city}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        >
                          <option value="">Select your city</option>
                          {citiesList.map((city) => (
                            <option key={city} value={city}>
                              {city}
                            </option>
                          ))}
                        </select>
                      </div>

                      {/* Gender */}
                      <div>
                        <label className="block mb-1">
                          Gender: <span className="text-red-500">*</span>
                        </label>
                        <select
                          required
                          name="gender"
                          value={userData.gender}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        >
                          <option value="">Select gender</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                          <option value="other">Other</option>
                        </select>
                      </div>

                      {/* Date of Birth */}
                      <div className="col-span-1 md:col-span-2">
                        <label className="block mb-1">
                          Date of Birth: <span className="text-red-500">*</span>
                        </label>
                        <input
                          required
                          type="date"
                          name="dob"
                          value={userData.dob}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        />
                      </div>

                      {/* Address */}
                      <div className="col-span-full">
                        <label className="block mb-1">
                          Address: <span className="text-red-500">*</span>
                        </label>
                        <input
                          required
                          type="text"
                          name="address"
                          placeholder="Enter your address"
                          value={userData.address}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        />
                      </div>

                      {/* State Dropdown */}
                      <div className="col-span-full">
                        <label className="block mb-1">
                          State: <span className="text-red-500">*</span>
                        </label>
                        <select
                          required
                          name="state"
                          value={userData.state}
                          onChange={handleChange}
                          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        >
                          <option value="">Select your state</option>
                          {statesList.map((state) => (
                            <option key={state} value={state}>
                              {state}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  <div className="w-full md:w-[200px] flex flex-col items-center mt-6 md:mt-0 md:ml-8">
                    <label className="cursor-pointer">
                      <input type="file" accept="image/*" className="hidden" />
                      <div className="w-32 h-32 rounded-full border-2 border-dashed border-gray-300 flex items-center justify-center hover:border-blue-500 transition-colors">
                        <span className="material-symbols-outlined text-gray-400">
                          add_a_photo
                        </span>
                      </div>
                    </label>
                    <span className="mt-2 text-sm text-gray-600">
                      Upload Photo
                    </span>
                  </div>
                </div>
              </section>

              {/* Educational Details Section */}
              <section
                id="educational"
                className="bg-white rounded-lg shadow-sm p-4 md:p-8"
              >
                <h2 className="text-2xl font-semibold mb-6">
                  Educational Details
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-4">
                  <div>
                    <label className="block mb-1">
                      Std: <span className="text-red-500">*</span>
                    </label>
                    <div className="w-full p-2 border rounded bg-gray-50 text-gray-700">
                      10th
                    </div>
                  </div>

                  {/* Board Type */}
                  <div>
                    <label className="block mb-1">
                      Board Type: <span className="text-red-500">*</span>
                    </label>
                    <select
                      required
                      name="boardType"
                      value={userData.boardType}
                      onChange={handleChange}
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                    >
                      <option value="">Select Board Type</option>                       
                      <option value="SSC">SSC</option>
                      <option value="CBSE">CBSE</option>
                      <option value="ICSE">ICSE</option>
                    </select>
                  </div>

                  {/* Maths Marks */}
                  <div>
                    <label className="block mb-1">
                      Maths: <span className="text-red-500">*</span>
                    </label>
                    <input
                      required
                      type="number"
                      name="maths"
                      placeholder="Enter your Maths marks"
                      value={userData.maths}
                      onChange={handleChange}
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                    />
                  </div>

                  {/* Science Marks */}
                  <div>
                    <label className="block mb-1">
                      Science: <span className="text-red-500">*</span>
                    </label>
                    <input
                      required
                      type="number"
                      name="science"
                      placeholder="Enter your Science marks"
                      value={userData.science}
                      onChange={handleChange}
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                    />
                  </div>

                  {/* Board Percentage */}
                  <div>
                    <label className="block mb-1">
                      Board Percentage: <span className="text-red-500">*</span>
                    </label>
                    <input
                      required
                      type="number"
                      name="boardPercentage"
                      placeholder="Enter %"
                      value={userData.boardPercentage}
                      onChange={handleChange}
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                    />
                  </div>
                </div>
              </section>

              {/* Hobbies Section */}
              <section
                id="interests"
                className="bg-white rounded-lg shadow-sm p-4 md:p-8"
              >
                <h2 className="text-2xl font-semibold mb-6">
                  Personal Interests and Hobbies
                </h2>
                <div className="space-y-4">
                  <div>
                    <label className="block mb-1">
                      Hobbies: <span className="text-red-500">*</span>
                    </label>
                    <div className="flex flex-col md:flex-row gap-4 items-start">
                      <div className="relative">
                        <button
                          onClick={() => setDropdownOpen(!dropdownOpen)}
                          className="w-full md:w-[200px] p-2 border rounded cursor-pointer hover:border-blue-500"
                        >
                          Select Hobbies
                        </button>
                        {dropdownOpen && (
                          <ul className="absolute w-full mt-1 bg-white border rounded shadow-lg z-10">
                            {[
                              "Reading",
                              "Coding",
                              "Cooking/Baking",
                              "Sports",
                              "Music",
                              "History/Geography",
                              "Dance",
                              "Photography",
                              "Drawing/Painting",
                              "Science",
                            ].map((hobby, index) => (
                              <li
                                key={index}
                                onClick={() => handleHobbySelect(hobby)}
                                className="p-2 hover:bg-blue-50 cursor-pointer"
                              >
                                {hobby}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {userData.hobbies.map((hobby, index) => (
                          <span
                            key={index}
                            className="px-3 py-1 bg-blue-50 rounded-full flex items-center gap-2"
                          >
                            {hobby}
                            <button
                              className="text-blue-600 hover:text-blue-800"
                              onClick={() => handleHobbyRemove(hobby)}
                            >
                              ×
                            </button>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Achievements */}
                  <div>
                    <label className="block mb-1">Achievements:</label>
                    <textarea
                      name="achievements"
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all min-h-[100px]"
                      placeholder="Enter your achievements"
                      value={userData.achievements}
                      onChange={handleChange}
                    />
                  </div>

                  {/* Certificates Button */}
                  <div>
                    <button
                      className="px-6 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                      onClick={(e) =>
                        document.getElementById("file-input").click()
                      } // Trigger file input click
                    >
                      Add Certificates
                    </button>

                    <input
                      type="file"
                      multiple
                      accept=".pdf,.doc,.docx"
                      id="file-input"
                      className="hidden"
                      onChange={handleFileUpload} // Handle file selection
                    />
                  </div>
                </div>
              </section>

              {/* Religion and Caste Section */}
              <section
                id="religion"
                className="bg-white rounded-lg shadow-sm p-4 md:p-8"
              >
                <h2 className="text-2xl font-semibold mb-6">
                  Religion and Caste
                </h2>
                <div className="flex flex-col md:flex-row gap-4">
                  {/* Religion Dropdown */}
                  <div className="flex-1">
                    <label className="block mb-1">Religion:</label>
                    <select
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                      value={userData.religion}
                      onChange={handleChange}
                      name="religion"
                    >
                      <option value="">Select Religion</option>
                      {religionsList.map((religion) => (
                        <option key={religion} value={religion}>
                          {religion}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Caste Dropdown */}
                  <div className="flex-1">
                    <label className="block mb-1">Caste:</label>
                    <select
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                      value={userData.caste}
                      onChange={handleChange}
                      name="caste"
                    >
                      <option value="">Select Caste</option>
                      {castesList.map((caste) => (
                        <option key={caste} value={caste}>
                          {caste}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </section>

              {/* Save and Back Buttons */}
              <div className="flex justify-center gap-4 pb-8">
                <button
                  onClick={() => navigate(-1)} // Go back to the previous page
                  className="px-6 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={handleValidationAndSave} // Validate and save
                  className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Userdetails;
