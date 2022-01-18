// Event handling
document.addEventListener("DOMContentLoaded",
  function () { 

    // Unobtrusive event binding
    document.querySelector("button")
      .addEventListener("click", function () {
        var name="";
        self=this;
        // Call server to get the name
        $ajaxUtils
          .sendGetRequest("data/file.txt", 
          
              function (request) {
                name = request.responseText;
                self.textContent=name;
        });

        document.querySelector("#content")
          .innerHTML = "<h2>Hello " + name + "!</h2>";
        
    });
});
