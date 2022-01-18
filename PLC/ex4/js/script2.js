document.addEventListener("DOMContentLoaded",
  function () {
    document.querySelector("button")
      .addEventListener("click", function () {
        var index = ["https://lemida.biu.ac.il/",
                    "https://ims.gov.il/",
                    "https://www.mizrahi-tefahot.co.il/",
                    "https://www.maariv.co.il/",
                    "https://www.wikipedia.org/"];
        (function loop(i) {
            if (i>= index.length) {
                return;
            }
            var url = index[i];
            $ajaxUtils.sendGetRequest(url, function (request) {
              var data = request.responseText;
              console.log('-->' + i + ' id: ' + data.substring(1,2000));
              console.log("---------------------------\n")
              loop(i + 1);
            });
        })(0);
  });
});