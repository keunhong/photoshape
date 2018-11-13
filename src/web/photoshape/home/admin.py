from django.contrib import admin
from .forms import UploadedImageForm
from .models import UploadedImage

# Register your models here.
class UploadedImageAdmin(admin.ModelAdmin):
	form = UploadedImageForm
	fields = ['title', 'original', 'mask']

admin.site.register(UploadedImage, UploadedImageAdmin)