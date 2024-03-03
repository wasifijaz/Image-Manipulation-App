from django.db import models

class Image(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')
    selected = models.BooleanField(default=False)

    def __str__(self):
        return self.title
