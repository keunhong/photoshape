from django.db import models

# Create your models here.
class Material(models.Model):
	mid = models.IntegerField(primary_key=True, unique=True)

class UploadedImage(models.Model):
    title = models.CharField(max_length=100, blank=True)
    original = models.ImageField(upload_to='original')
    mask = models.ImageField(upload_to='mask')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    computed_materials = models.ManyToManyField(Material, related_name='computed', blank=True)
    user_material = models.ForeignKey(Material, blank=True, related_name='user', null=True, on_delete=models.CASCADE)
