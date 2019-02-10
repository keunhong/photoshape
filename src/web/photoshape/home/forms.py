from django import forms
from .models import UploadedImage

class UploadedImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ('title', 'original')

class UploadedImageForm2(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ('title', 'original', 'mask')
