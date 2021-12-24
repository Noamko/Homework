// Noam Koren 308192871
// Yuval Khanimov 318970902
function add() {
    let num1 = document.getElementById("num1");
    let num2 = document.getElementById("num2");
    var result = document.getElementById("lblResult");
    result.innerHTML =  (parseFloat(num1.value) +  parseFloat(num2.value))
}