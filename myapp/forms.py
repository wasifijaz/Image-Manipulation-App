from django import forms

class ImageUploadForm(forms.Form):
    title = forms.CharField(max_length=100)
    image = forms.ImageField()

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            pass
        return image

class ClusterForm(forms.Form):
    cluster_input = forms.IntegerField(max_value=10)
