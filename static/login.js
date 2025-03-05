import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-analytics.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";

// Your web app's Firebase configuration
const firebaseConfig = {
   apiKey: "AIzaSyDY32CjxqTnsUlmLSg_JDoYKaO5nSqMcKY",
   authDomain: "mockup-3b1d2.firebaseapp.com",
   projectId: "mockup-3b1d2",
   storageBucket: "mockup-3b1d2.appspot.com",
   messagingSenderId: "1000372772813",
   appId: "1:1000372772813:web:4ef25d6ad044c954e06e84",
   measurementId: "G-2Y830SVFWB"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// Initialize Firebase Auth
const auth = getAuth(app);

// Get the submit button and add event listener
const submit = document.getElementById('login-btn'); // Change this to 'login-btn'
submit.addEventListener("click", function (event) {
    event.preventDefault(); // Prevent form submission

    // Retrieve email and password values inside the event listener
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    signInWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            // Signed in 
            const user = userCredential.user;
            alert("Login successfully");
            window.location.href = "homepage.html"; // Redirect to the home page or desired page
        })
        .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            alert(errorMessage); // Display the error message
        });
});
