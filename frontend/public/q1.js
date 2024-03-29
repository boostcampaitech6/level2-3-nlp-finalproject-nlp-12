// import { uid } from "./index.js";
// import { age } from "./index.js";
// import { gender } from "./index.js";
// import { baseUrl } from "./index.js"

// https://os94.tistory.com/125


// Retrieve the unique identifier from sessionStorage
const userID = localStorage.getItem('uid');
console.log(userID);
if (!userID) {
    window.location.href = '/error';
}


const baseUrl = 'http://175.45.194.237:8000/predict?';
const age = localStorage.getItem('age');
const gender = localStorage.getItem('gender');
const uid = localStorage.getItem('uid');
const key = localStorage.getItem('key');
console.log(age, gender, uid);

const requestData = {
    id : uid,
    age : age,
    gender : gender,
    question : 1,
    created_at : get_nowtime(),
    key : key,
};

var audioBlob;

async function next_question() {
    try {
        const formData = new FormData();
        formData.append('audio_file', audioBlob);
        console.log("fomrdata ready")

        const apiUrl = baseUrl + "id=" + requestData["id"] + "&" + "age=" + requestData["age"] + "&" + "gender=" + requestData["gender"] + "&" + "question=" + requestData["question"] + "&" + "created_at=" + requestData["created_at"] + "&" + "key=" + requestData["key"];
        console.log(apiUrl)
    
        await fetch(apiUrl, {
            method: 'POST',
            headers: {},
            body: formData,
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                console.log("Request Done.");
                return response.json();
            })
            .then(data => {
                console.log('Audio upload successful:', data);
            })
            .catch(error => {
                console.error('Error uploading audio:', error);
            });
        
        window.location.href = '/q2';

    } catch (error) {
        console.error('Error uploading audio:', error);
        window.location.href = '/q2';
    }
}


function get_nowtime() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    // const hours = String(now.getHours()).padStart(2, '0');
    // const minutes = String(now.getMinutes()).padStart(2, '0');
    // const seconds = String(now.getSeconds()).padStart(2, '0');
    const formattedDateTime = `${year}${month}${day}`;
    return formattedDateTime;
}


let mediaRecorder;

const startRecordingButton = document.getElementById('startRecording');
const stopRecordingButton = document.getElementById('stopRecording');
const audioPlayback = document.getElementById('audioPlayback');

startRecordingButton.addEventListener('click', startRecording);
stopRecordingButton.addEventListener('click', stopRecording);

function startRecording() {

    let recordedChunks = [];

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        startRecordingButton.disabled = true;
        stopRecordingButton.disabled = false;

        mediaRecorder.ondataavailable = event => {
            recordedChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
            // uploadAudio(audioBlob);
            const audioUrl = URL.createObjectURL(audioBlob);
            console.log(audioUrl);
            audioPlayback.src = audioUrl;
        };
        })
        .catch(error => {
        console.error('Error accessing microphone:', error);
        });
}

function stopRecording() {
    mediaRecorder.stop();
    startRecordingButton.disabled = false;
    stopRecordingButton.disabled = true;
}

function redirectToMain() {
    window.location.href = '/';
}