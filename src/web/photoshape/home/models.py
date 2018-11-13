from django.db import models

# Create your models here.

class UploadedImage(models.Model):
    title = models.CharField(max_length=100, blank=True)
    original = models.ImageField(upload_to='original')
    mask = models.ImageField(upload_to='mask')
    uploaded_at = models.DateTimeField(auto_now_add=True)