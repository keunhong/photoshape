var BASE_URL = 'http://localhost:9999/terial/evaluation';

var submittedShapeIds = [];
var currentShapeId = -1;

var currentPairIds = [];


var tmpl = $.templates(`
  <h3>{{:data.shape.id }}</h3>
  <h4>{{:numDone}} done this session!</h4>
  <table class="result-table table table-striped table-sm">
    {{for data.pairs}}
    <tr>
    <th>Exemplar</th>
    <th>Rendering</th>
    <th></th>
    </tr>
    <tr>
      <td class="img-td">
        <a href="{{:exemplar_url}}" data-toggle="lightbox">
          <img class="exemplar-img" src="{{:exemplar_url}}">
        </a>
      </td>
      <td class="img-td">
        <a href="{{:rend_url}}" data-toggle="lightbox">
          <img class="result-img" src="{{:rend_url}}">
        </a>
      </td>
      <td class="result-control-td">
      
      <div class="form-group">
        <button class="annotate-btn btn btn-outline-success" data-pair-id="{{:pair_id}}" data-category="good">Good</button>
        <button class="annotate-btn btn btn-outline-primary" data-pair-id="{{:pair_id}}" data-category="acceptable">Acceptable</button>
        <button class="annotate-btn btn btn-outline-danger" data-pair-id="{{:pair_id}}" data-category="wrong_material">Failure: Material</button>
        <button class="annotate-btn btn btn-outline-danger" data-pair-id="{{:pair_id}}" data-category="wrong_ambiguous">Failure: Ambiguous</button>
        <button class="annotate-btn btn btn-outline-danger" data-pair-id="{{:pair_id}}" data-category="wrong_color">Failure: Color</button>
        <button class="annotate-btn btn btn-outline-danger" data-pair-id="{{:pair_id}}" data-category="wrong_alignment">Failure: Alignment</button>
        <button class="annotate-btn btn btn-outline-danger" data-pair-id="{{:pair_id}}" data-category="under_segmented">Failure: Under Segmented</button>
        <button class="annotate-btn btn btn-outline-danger" data-pair-id="{{:pair_id}}" data-category="over_segmented">Failure: Over Segmented</button>
        <button class="annotate-btn btn btn-outline-secondary" data-pair-id="{{:pair_id}}" data-category="bad_exemplar">Bad Exemplar</button>
        <button class="annotate-btn btn btn-outline-secondary" data-pair-id="{{:pair_id}}" data-category="bad_shape">Bad Shape</button>
        <button class="annotate-btn btn btn-outline-secondary" data-pair-id="{{:pair_id}}" data-category="limitation">Limitation</button>
        <button class="annotate-btn btn btn-outline-secondary" data-pair-id="{{:pair_id}}" data-category="not_sure">Not Sure</button>
      </td>
    </tr>
    {{/for}}
  </table>
  `);


$(document).ready(function () {

  $('#prev-shape-btn').click(function () {
    if (submittedShapeIds.length === 0) {
      return;
    }

    currentPairIds = [];
    $.ajax({
      method: 'GET',
      url: BASE_URL + '/shape',
      data: {
        'username': username,
        'token': token,
        'shape_source': shapeSource,
        'result_set': resultSet,
        'shape_id': submittedShapeIds.pop()
      }
    }).done(function (data) {
      console.log(data);
      $('#result-div').html(tmpl.render({
        data: data,
        numDone: submittedShapeIds.length
      }));
      currentShapeId = data['shape']['id'];
      for (pair of data['pairs']) {
        currentPairIds.push(pair['pair_id']);
      }
      console.log(submittedShapeIds);
      updateEventListeners();
    }).fail(function (data) {
      alert('Error: ' + data.responseText);
      console.error(data);
    });
  })

  loadNewShape();

  // $('#next-shape-btn').click(function () {
  //   if (currentPairIds.length > 0) {
  //     alert('Please finish current set first');
  //     return;
  //   }
  //   loadNewShape();
  // });
});


function updateEventListeners() {
  $('.annotate-btn').click(postResult);
}


function postResult(e) {
  let pairId = $(this).data('pairId');
  let category = $(this).data('category');

  $.ajax({
    method: 'POST',
    url: BASE_URL + '/annotation',
    data: {
      'username': username,
      'token': token,
      'shape_source': shapeSource,
      'result_set': resultSet,
      'pair_id': pairId,
      'category': category,
    }
  }).done(function (data) {
    $("button[data-pair-id='" + pairId + "']").each(function (idx, el) {
      $(el).removeClass('active');
    });
    let activeButton = $($("button[data-pair-id='" + pairId + "'][data-category='" + category + "']")[0]);
    activeButton.addClass('active');
    currentPairIds = currentPairIds.filter(item => item !== pairId);
    console.log(data);
    console.log('currentPairIds', currentPairIds);
    if (currentPairIds.length === 0) {
      loadNewShape();
    }
  }).fail(function (data) {
    alert('Error: ' + data.responseText);
    console.error(data);
  });
}


function loadNewShape() {
  $.ajax({
    method: 'GET',
    url: BASE_URL + '/shape',
    data: {
      'username': username,
      'token': token,
      'shape_source': shapeSource,
      'result_set': resultSet
    }
  }).done(function (data) {
    console.log(data);
    $('#result-div').html(tmpl.render({
      data: data,
      numDone: submittedShapeIds.length
    }));
    if (currentShapeId !== -1) {
      submittedShapeIds.push(currentShapeId);
    }

    for (pair of data['pairs']) {
      currentPairIds.push(pair['pair_id']);
    }

    currentShapeId = data['shape']['id'];
    console.log(submittedShapeIds);
    updateEventListeners();
  }).fail(function (data) {
    alert('Error: ' + data.responseText);
    console.error(data);
  });
}


