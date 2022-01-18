// Noam Koren 308192871
exports.ajaxInit= function () {
    var XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;
    var ajaxUtils = {};
    function getRequestObject() {
        if (XMLHttpRequest) {
          return (new XMLHttpRequest());
        } 
        else if (ActiveXObject) {
          // For very old IE browsers (optional)
          return (new ActiveXObject("Microsoft.XMLHTTP"));
        } 
        else {
          alert("Ajax is not supported!");
          return(null); 
        }
    }
    ajaxUtils.sendGetRequest = 
      function(requestUrl, responseHandler) {
        var request = getRequestObject();
        request.onreadystatechange = 
          function() { 
            handleResponse(request, responseHandler); 
          };
        request.open("GET", requestUrl, true);
        request.send(null); // for POST only
      };
    function handleResponse(request,
                            responseHandler) {
      if ((request.readyState == 4) &&
         (request.status == 200)) {
        responseHandler(request);
      }
    }
      global.$ajaxUtils = ajaxUtils;
}