$(document).ready(function () {
  $('.shape-exclude-checkbox').change(function () {
    let el = $(this);
    let checked = $(this).is(':checked');
    let shapeId = $(this).data('shape-id');

    putShapeExcluded(
        shapeId,
        checked,
        function (data) {
          console.log(data);
          if (data.status === 'success') {
            if (data['shape']['exclude']) {
              el.closest('.card').addClass('card-disabled');
            } else {
              el.closest('.card').removeClass('card-disabled');
            }
          }
        },
        function (data) {
          alert('Error: ' + data.responseText);
          console.error(data);
        });
  });
})