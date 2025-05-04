📘 DishaDoot - A Career Navigation Platform

DishaDoot is a career guidance platform tailored for 10th-pass students. The system helps users select suitable career paths based on their hobbies and mental abilities. It provides personalized course recommendations offered by Mumbai University (MU) and suggests applicable scholarships.

🌟 Features

🎯 Recommends career paths and MU courses based on user's hobbies and aptitude.
🧠 Uses a fine-tuned BERT model for intelligent recommendations.
🎓 Suggests scholarships aligned with selected courses.
🔐 Firebase Authentication
📋 Quiz-based aptitude test
🔄 Real-time course suggestions
📦 Integrated frontend-backend architecture

------------------------------------------------------------------------------------------------------------------------------------------

🚀 Getting Started

1️⃣ Clone & Extract Project

* Download the ZIP file of this repository.
* Extract the content to your desired location.

2️⃣ Setup the Backend

Navigate to the `backend/` folder:

``` cd backend ```

Install backend dependencies:

``` pip install -r requirements.txt```

Now, run the model to generate the `bert_recommendation.pt` file:

``` python run_model.py ```

⚠️ After the model file is generated, you can stop this process (close the terminal or interrupt it).

3️⃣ Setup Firebase

* Go to [Firebase Console](https://firebase.google.com/).
* Create a new project and generate Web App credentials.
* Replace the placeholder content inside `frontend/firebase.js` with your Firebase configuration.

4️⃣ Setup the Frontend

Navigate to the project root:

``` cd frontend ```

Initialize the project using Vite:

``` npm create vite@latest ```

Install required dependencies:

```
npm install
npm install firebase axios screenfull react-toastify react-router-dom
```

📦 Dependency Management

Python (Backend)

All Python dependencies are listed in `backend/requirements.txt`. Install them using:

``` pip install -r requirements.txt ```



🔧 Running the Project

Open three separate terminals and follow the steps below:

Terminal 1: Frontend

```
cd frontend
npm run dev
```

Terminal 2: Backend Quiz API

```
cd backend
python server.py
```

Terminal 3: Course Recommendation Model

```
cd backend
python recommend.py
```

All three processes should be running simultaneously.

📬 Support

If you run into any issues, feel free to reach out via Issues or contact the project maintainer.

📜 License

This project is for educational purposes .


