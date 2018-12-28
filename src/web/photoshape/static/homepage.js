// var el = document.getElementById('upload-image');
// var vanilla = new Croppie(el, {
//     viewport: { width: 300, height: 300 },
//     showZoomer: false,
// });
var outlineImage = new Image();
var croppedImage;
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

function result(data){
  ids = data['materials']
  url = data['url']
  fid = data['form']
  text = "<h2>Your image:</h2><img id='original' src=" + url + " alt='your image' />"
  $('#original').html(text)
  var results = document.getElementById("infer_results");
  var title = document.createElement("h2")
  title.textContent = "Results"
  results.appendChild(title);

  for (var i = 0; i < ids.length; i++) {
    id = ids[i][0]
    name = ids[i][1]
    var res_container = document.createElement("div")
    res_container.className = "res_container"
    var res_label = document.createElement("label")
    res_label.htmlFor = id
    res_label.innerHTML = name
    var res_img = document.createElement("img")
    res_img.className = "res_img"
    res_img.src = "/images/materials/" + id +"/images/previews/bmps.png"
    res_img.id = id
    var res_input = document.createElement("input")
    res_input.type = "radio"
    res_input.name = "result"
    res_input.value = id
    res_container.appendChild(res_label);
    res_container.appendChild(res_img);
    res_container.appendChild(res_input);
    results.appendChild(res_container);
  }
  var form_id = document.createElement("input")
  form_id.name = "form_id"
  form_id.type = "hidden"
  form_id.value = fid
  var submit_btn = document.createElement("input")
  submit_btn.type = "submit"
  submit_btn.style = "display:none"
  results.appendChild(form_id)
  results.appendChild(submit_btn)  
  $('#submit_btn').show();
}

function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();
      $('#cropped').html("");
      $('#crop-btn').show();
      $('#upload-image').attr('src', e.target.result);
      $('#file-upload-content').show();
      $('#image-title-wrap').html(input.files[0].name);
      $('#submit_btn').hide();
      $('#original').html("");
      $('#error').hide();
      document.getElementById("infer_results").innerHTML = "";
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
              type: 'blob',
              size: {width: 224, height: 224}
          }).then(function (blob) {
            croppedImage = blob;
    });
  $uploadCrop.croppie('result', {
              type: 'rawcanvas',
              size: 'viewport',
          }).then(function (rawcanvas) {
    $('#file-upload-content').hide();
    $('#cropped').html(rawcanvas);
    $('#controls').show();
    $('#crop-btn').hide();
    $(rawcanvas).attr('id', 'canvas');
    outlineImage.src = rawcanvas.toDataURL('image/png');
    init();
    // outlineImage.src = rawcanvas.toDataURL('image/png');
    // $('#canvasDiv').show()
    // redraw();

  });

}

function submitResult() {
   document.forms[0].submit();
}

function infer() {
    var fd = new FormData();
    var filename = document.getElementById('image-title-wrap').innerHTML;
    drawEmpty();
     document.getElementById('canvas').toBlob(function(blob) {
        maskImage = blob;

    fd.append('original', croppedImage, filename);
    fd.append('mask', maskImage, "mask-"+filename);
    fd.append('csrfmiddlewaretoken', getCookie('csrftoken'));
    $('#load').show();
    var tet = $.ajax({
        url: '/results/',
        type: 'POST',
        data: fd,
        async: true,
        contentType: false,
        enctype: 'multipart/form-data',
        processData: false,
        success: function (response) {
          $('#file-upload-content').hide();
          data = JSON.parse(response)
          $('#load').hide();
          result(data)
        },
        error: function (error) {
          $('#load').hide();
          $('#error').html(error);
          $('#error').show();
        }
    }).responseText;

      }, 'image/png', 1);
    $('#cropped').html("");
    $('#controls').hide();

}


var canvas, ctx,
    brush = {
        x: 0,
        y: 0,
        color: '#ffffff',
        size: 20,
        down: false,
    },
    strokes = [],
    currentStroke = null;

function drawEmpty() {
    ctx.clearRect(0, 0, canvas.width(), canvas.height());
    ctx.lineCap = 'round';
    for (var i = 0; i < strokes.length; i++) {
        var s =strokes[i];
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.size;
        ctx.beginPath();
        ctx.moveTo(s.points[0].x, s.points[0].y);
        for (var j = 0; j < s.points.length; j++){
            var p = s.points[j];
            ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
    }
}

function redraw () {
    ctx.clearRect(0, 0, canvas.width(), canvas.height());
    ctx.drawImage(outlineImage, 0, 0, canvas.width(), canvas.height());
    ctx.lineCap = 'round';
    for (var i = 0; i < strokes.length; i++) {
        var s =strokes[i];
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.size;
        ctx.beginPath();
        ctx.moveTo(s.points[0].x, s.points[0].y);
        for (var j = 0; j < s.points.length; j++){
            var p = s.points[j];
            ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
    }

}

outlineImage.onload=function() {
  ctx.drawImage(outlineImage, 0, 0, canvas.width(), canvas.height());
}

function init (url) {
    canvas = $('#canvas');
    // canvas = document.getElementById('canvas').getContext("2d")
    canvas.attr({
        width: 404,
        height: 404,
    });
    ctx = canvas[0].getContext('2d');
    strokes = [];
    function mouseEvent (e){
        ctx_rect= ctx.canvas.getBoundingClientRect();
        brush.x = e.clientX - ctx_rect.left;
        brush.y = e.clientY - ctx_rect.top;

        currentStroke.points.push({
            x: brush.x,
            y: brush.y,

        });
        redraw();
    }

    canvas.mousedown(function (e){
        brush.down = true;

        currentStroke = {
            color: brush.color,
            size: brush.size,
            points: [],
        };
        strokes.push(currentStroke);

        mouseEvent(e);
    }) .mouseup(function (e) {
        brush.down = false;

        mouseEvent(e);

        currentStroke = null;
    }) .mousemove(function (e) {
        if (brush.down)
            mouseEvent(e);
    }) .mouseleave(function (e) {
        brush.down = false;
    });


    $('#undo-btn').click(function (){
        strokes.pop();
        redraw();
    });
    $('#clear-btn').click(function (){
        strokes = [];
        redraw();
    });

    $('#color-picker').on('input', function () {
        brush.color = this.value;
    });
    $('#brush-size').on('input', function () {
        brush.size = this.value;
    });
    }
