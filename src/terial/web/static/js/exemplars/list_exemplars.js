

$(document).ready(function () {
  $('.exemplar-exclude-checkbox').change(function () {
    let el = $(this);
    let checked = $(this).is(':checked');
    let exemplarId = $(this).data('exemplar-id');

    putExemplarExcluded(
        exemplarId,
        checked,
        function() {
          console.log(data);
          if (data.status === 'success') {
            el.parent().append($('<p>OK</p>'));
          } else {
            el.parent().append($('<p>Hmm..</p>'));
          }
        },
        function () {
          alert('Error: ' + data.responseText);
          console.error(data);
        });
  });
});
