// Noam Koren
// 308192871

// use this to run in chrome
(function(global) {
    var counter = 0;
    function make () {
        return ++counter;
    }
    global.myCounter = make
})(window);


// use this to run in node js

// (function(global) {
//     var counter = 0;
//     function make () {
//         return ++counter;
//     }
//     global.myCounter = make
// })(global);


console.log(myCounter())
console.log(myCounter())
console.log(myCounter())
counter = 0;
console.log(myCounter())
console.log(myCounter())
console.log(myCounter())