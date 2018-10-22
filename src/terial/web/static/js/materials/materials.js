
function putMaterialScale(materialId, scale, onsuccess, onfail) {
  $.ajax({
    method: 'PUT',
    url: BASE_URL + '/materials/' + materialId + '/default_scale',
    data: {
      'default_scale': scale
    }
  }).done(function (data) {
    if (data.status === 'success') {
      onsuccess(data);
    } else {
      onfail(data);
    }
  }).fail(function (data) {
    alert('Error: ' + data.responseText);
    console.error(data);
  });
}


function putMaterialEnabled(materialId, enabled, onsuccess, onfail) {
  $.ajax({
    method: 'PUT',
    url: BASE_URL + '/materials/' + materialId + '/enabled',
    data: {
      'enabled': enabled
    }
  }).done(function (data) {
    if (data.status === 'success') {
      onsuccess(data);
    } else {
      onfail(data);
    }
  }).fail(onfail);
}
