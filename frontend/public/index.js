function generateRandomNumber() {
    // Generate a random number with 12 digits
    const min = Math.pow(10, 11); // Minimum 12-digit number (100000000000)
    const max = Math.pow(10, 12) - 1; // Maximum 12-digit number (999999999999)
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function Start() {
    
    // 나이 파싱
    const age = document.getElementById("select_age").value;
    // 성별 파싱
    const gender = document.getElementById("select_gender").value;
    
    // todo 추후 주석처리할것
    // key = document.getElementById("key").value;
    const key = ""; // 여기에 key값 입력 후 실행
    
    // todo : 세션 ID를 통해 UID 발급
    // 불러올 게 없으면 새로 발급
    const uid = generateRandomNumber();
    localStorage.setItem('uid', uid.toString());
    
    console.log(age, gender, uid);

    localStorage.setItem('age', age);
    if (gender == "male") {
        localStorage.setItem('gender', 1);
    }
    else {
        localStorage.setItem('gender', 0);
    }
    localStorage.setItem('uid', uid);
    localStorage.setItem('key', key);

    // 다음 페이지로 넘김
    window.location.href = '/q1';

}

function redirectToMain() {
    window.location.href = '/';
}