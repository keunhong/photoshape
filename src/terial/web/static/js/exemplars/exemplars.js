

function putExemplarExcluded(exemplarId, excluded, onsuccess, onfail) {
  $.ajax({
    method: 'PUT',
    url: BASE_URL + '/exemplars/' + exemplarId + '/exclude',
    data: {
      'exclude': excluded
    }
  }).done(onsuccess).fail(onfail);
}
