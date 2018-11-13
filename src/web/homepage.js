// var el = document.getElementById('upload-image');
// var vanilla = new Croppie(el, {
//     viewport: { width: 300, height: 300 },
//     showZoomer: false,
// });
var outlineImage = new Image();
$uploadCrop = $('#upload-image').croppie({
        viewport: {
            width: 400,
            height: 400,
        },
    });

    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');
    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();
      // $('#cropped').html("");
      $('#canvasDiv').hide();
      $('#upload-image').attr('src', e.target.result);
      $('#file-upload-content').show();
      $('#image-title-wrap').html(input.files[0].name);
      // vanilla.bind({
      //     url: e.target.result,
      // });
                $uploadCrop.croppie('bind', {
                    url: e.target.result
                });

    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

  function showCropped() {

  // vanilla.result('blob','viewport', 'jpeg').then(function(blob) {
    //console.log(canvas.toDataURL('image/png'));
    // var imageData = canvas.toDataURL('image/png');
    // document.getElementById("file").value = imageData;

    

    // console.log(document.getElementById("file").value);

  $uploadCrop.croppie('result', {
              type: 'rawcanvas',
              size: 'viewport',
          }).then(function (rawcanvas) {
    $('#file-upload-content').hide();
    // $('#cropped').html(rawcanvas);
    // $(canvas).attr('id', 'canvas');
    redraw();
    outlineImage.src = rawcanvas.toDataURL('image/png');
    $('#canvasDiv').show()
    redraw();

  });


  // $uploadCrop.croppie('result', {
  //             type: 'blob',
  //             size: {width: 224, height: 224}
  //         }).then(function (blob) {
  //   var fd = new FormData();
  //   var filename = document.getElementById('image-title-wrap').innerHTML;
  //   fd.append('file', blob, filename);
  //   fd.append('csrfmiddlewaretoken', getCookie('csrftoken'));
  //   var tet = $.ajax({
  //       type: 'POST',
  //       data: fd,
  //       async: true,
  //       contentType: false,
  //       enctype: 'multipart/form-data',
  //       processData: false,
  //       success: function (response) {
  //         $('#file-upload-content').hide();
  //       },
  //       error: function (error) {
  //           console.log(error);
  //       }
  //   }).responseText;
  // });


}

var canvasDiv = document.getElementById('canvasDiv');
canvas = document.createElement('canvas');
canvas.setAttribute('width', 404);
canvas.setAttribute('height', 404);
canvas.setAttribute('id', 'canvas');
canvasDiv.appendChild(canvas);
if(typeof G_vmlCanvasManager != 'undefined') {
  canvas = G_vmlCanvasManager.initElement(canvas);
}
context = canvas.getContext("2d");
$('#canvasDiv').hide();

$('#canvas').mousedown(function(e){
  var mouseX = e.pageX - this.offsetLeft;
  var mouseY = e.pageY - this.offsetTop;
    
  paint = true;
  addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
  redraw();
});
$('#canvas').mousemove(function(e){
  if(paint){
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
    redraw();
  }
});
$('#canvas').mouseup(function(e){
  paint = false;
});
$('#canvas').mouseleave(function(e){
  paint = false;
});
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;

function addClick(x, y, dragging)
{
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}
    function clearCanvas() {

      context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    }

function redraw(){
  console.log("redraw is called");
  context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
  
  context.strokeStyle = "#df4b26";
  context.lineJoin = "round";
  context.lineWidth = 5;
  context.drawImage(outlineImage, 0, 0, 404, 404);
      
  for(var i=0; i < clickX.length; i++) {    
    context.beginPath();
    if(clickDrag[i] && i){
      context.moveTo(clickX[i-1], clickY[i-1]);
     }else{
       context.moveTo(clickX[i]-1, clickY[i]);
     }

     context.lineTo(clickX[i], clickY[i]);
     context.closePath();
     context.stroke();

  }
}