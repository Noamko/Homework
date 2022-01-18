// Event handling
document.addEventListener("DOMContentLoaded",
  function () { 

    // Unobtrusive event binding
    document.querySelector("button")
      .addEventListener("click", function () {
        var name="";

        // Call server to get the name
        $ajaxUtils
          .sendGetRequest("data/file.txt", 
            
              function (request) {
                name = request.responseText;
              
        });

        document.querySelector("div")
          .innerHTML = "<h2>Hello " + name + "!</h2>";
        
    });
});

