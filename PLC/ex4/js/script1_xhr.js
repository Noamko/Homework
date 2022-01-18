// Event handling onreadystatechange onload
document.addEventListener("DOMContentLoaded",
  function () { 

    // Unobtrusive event binding
    document.querySelector("button")
      .addEventListener("click", function () {
        var name="";

        // Call server to get the name
        var request = new XMLHttpRequest();
        request.open("GET", "data/file.txt", true);
        request.onreadystatechange = function() { 
            console.log(request.readyState,request.status)
            if(request.readyState === 4 && request.status === 200) {
                name = request.responseText;
                document.querySelector("div")
                  .innerHTML = "<h2>Hello " + name + "!</h2>";
                
            }
        }
        request.send();
        setTimeout(function(){console.log(fib(42))},100)
    });
});


