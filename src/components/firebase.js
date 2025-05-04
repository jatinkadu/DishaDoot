// src/firebase.js
import { initializeApp } from 'firebase/app';
import { getAuth} from 'firebase/auth';
import { getFirestore,collection, addDoc, query, orderBy, onSnapshot } from "firebase/firestore";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "XXXXXXXXXXXXXXXXXXXXXXX",
  authDomain: "XXXXXXXXXXXXXXXXXXX",
  projectId: "XXXXXXXXXXXXXXXXXXXXX",
  storageBucket: "XXXXXXXXXXXXXXXXX",
  messagingSenderId: "XXXXXXXXXXXXX",
  appId: "XXXXXXXXXXXXXXXXXXXXXXXXX"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app); 
const messagesCollection = collection(db, "chat_messages");

export { auth,db,storage,messagesCollection, addDoc, query, orderBy, onSnapshot, collection};
export const apiKey = "XXXXXXXXXXXXXXXXXXXX"; // Replace with your real API key
