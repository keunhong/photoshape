

$(document).ready(function() {
  $('.material-scale-edit-btn').click(function() {
    let el = $(this);
    let materialId = el.data('material-id');
    let inputEl = $('#material-' + materialId + '-scale-input');
    let scale = inputEl.val();

    putMaterialScale(
        materialId,
        scale,
        function(data) {
          inputEl.css('background-color', '#34af5d');
        },
        function(data) {
          console.log(data);
        });
  });

  $('.material-enabled-checkbox').change(function() {
    let el = $(this);
    let checked = $(this).is(':checked');
    let materialId = $(this).data('material-id');

    putMaterialEnabled(
        materialId,
        checked,
        function (data) {
          console.log(data);
          if (data.status === 'success') {
            if (data['material']['enabled']) {
              el.closest('.card').removeClass('card-disabled');
            } else {
              el.closest('.card').addClass('card-disabled');
            }
          }
        },
        function (data) {
          alert('Error: ' + data.responseText);
          console.error(data);
        });
  });
});
