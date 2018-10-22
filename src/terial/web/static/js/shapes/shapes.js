
function putShapeExcluded(shapeId, excluded, onsuccess, onfail) {
  $.ajax({
    method: 'PUT',
    url: BASE_URL + '/shapes/' + shapeId + '/exclude',
    data: {
      'exclude': excluded
    }
  }).done(onsuccess).fail(onfail);
}
