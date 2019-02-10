from django.contrib import admin
from .forms import UploadedImageForm
from .models import UploadedImage, Material

# Register your models here.
class UploadedImageAdmin(admin.ModelAdmin):
	form = UploadedImageForm
	fields = ['title', 'original', 'computed_materials', 'user_material']

admin.site.register(UploadedImage, UploadedImageAdmin)
admin.site.register(Material)
