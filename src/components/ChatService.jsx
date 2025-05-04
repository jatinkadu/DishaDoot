import axios from "axios";
import { apiKey } from "./firebase"; // Store your Gemini API key separately

const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`; // Replace with your actual API URL
const cache = new Map();

export const sendMessageToGemini = async (userMessage, retries = 3, delay = 3000) => {
  if (cache.has(userMessage)) {
    console.log("✅ Returning cached response for:", userMessage);
    return cache.get(userMessage);
  }

  try {
    const response = await axios.post(
      API_URL,
      {
        contents: [{ role: "user", parts: [{ text: userMessage }] }],
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `${apiKey}`, // Use the API key in the Authorization header if required
        },
      }
    );

    let botReply = response.data.candidates?.[0]?.content?.parts?.[0]?.text || "Sorry, I couldn't understand.";
    botReply = botReply.replace(/\*\*\*/g, "").trim();

    cache.set(userMessage, botReply); // Store response in cache to prevent duplicate API calls
    return botReply;
  } catch (error) {
    if (error.response?.status === 429 && retries > 0) {
      console.warn(`⚠️ Rate limit reached! Retrying in ${delay / 1000} seconds...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return sendMessageToGemini(userMessage, retries - 1, delay * 2);
    }

    console.error("❌ Error fetching response from Gemini:", error);
    return "Error getting response.";
  }
};

