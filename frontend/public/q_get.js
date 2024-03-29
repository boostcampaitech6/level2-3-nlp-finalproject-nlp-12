
const baseUrl = 'http://175.45.194.237:8000/predict/';
const uid = localStorage.getItem('uid');
const key = localStorage.getItem('key');

async function fetchData() {

    const apiURL = baseUrl + uid + "?" + "key=" + key;

    try {
        const response = await fetch(apiURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        console.log('Data:', data);
        localStorage.setItem('result_data', JSON.stringify(data));
        window.location.href = '/result'
        
        // todo 데이터 저장
        }
        catch (error) {
            console.error('There was a problem with the fetch operation:', error);
            throw error;
        }
    }

window.onload = fetchData;
    
// // Call the async function
// (async () => {
//     try {
//     const data = await fetchData();
//     // Handle the response data here
//     } catch (error) {
//     // Handle errors here
//     }
// })();


function redirectToMain() {
    window.location.href = '/';
}
