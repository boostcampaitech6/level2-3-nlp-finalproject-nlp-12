
const userID = localStorage.getItem('uid');
console.log(userID);
if (!userID) {
    window.location.href = '/error';
}

const data = JSON.parse(localStorage.getItem('result_data'));
console.log(data);
console.log(data["result"]);
console.log(data["result"][0]);
console.log(data["result"][0]["question"]);
console.log(data["result"][0]["prob"]);

// for (let key in data) {
//     if (data.hasOwnProperty(key)) {
//         console.log(key + ': ' + data[key]);
//     }
// }

const q1_prob = document.getElementById('q1_prob');
const q2_prob = document.getElementById('q2_prob');
const q3_prob = document.getElementById('q3_prob');
const q4_prob = document.getElementById('q4_prob');
const q5_prob = document.getElementById('q5_prob');
const qall_prob = document.getElementById('qall_prob');

q1_prob.textContent = Math.round(data["result"][0]["prob"]*100)/100;
q2_prob.textContent = Math.round(data["result"][1]["prob"]*100)/100;
q3_prob.textContent = Math.round(data["result"][2]["prob"]*100)/100;
q4_prob.textContent = Math.round(data["result"][3]["prob"]*100)/100;
q5_prob.textContent = Math.round(data["result"][4]["prob"]*100)/100;
qall_prob.textContent = Math.round(((data["result"][0]["prob"] + data["result"][1]["prob"] + data["result"][2]["prob"] + data["result"][3]["prob"] + data["result"][4]["prob"])/5)*100);

function redirectToMain() {
    window.location.href = '/';
}



const ctx = document.getElementById('myChart').getContext('2d');
const myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['행복', '불행', '그림-긍정', '그림-중립', '그림-부정'],
        datasets: [{
        label: 'Logits',
        data: [q1_prob.textContent, q2_prob.textContent, q3_prob.textContent, q4_prob.textContent, q5_prob.textContent],
        backgroundColor: [
            'rgba(255, 99, 132, 0.2)',
            'rgba(54, 162, 235, 0.2)',
            'rgba(255, 206, 86, 0.2)',
            'rgba(75, 192, 192, 0.2)',
            'rgba(153, 102, 255, 0.2)',
            // 'rgba(255, 159, 64, 0.2)'
        ],
        borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            // 'rgba(255, 159, 64, 1)'
        ],
        borderWidth: 1
        }]
    },
    options: {
        scales: {
        y: {
            beginAtZero: true
        }
        }
    }
});