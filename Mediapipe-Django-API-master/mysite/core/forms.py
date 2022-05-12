from django import forms

from .models import Image

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = ('title', 'image')

class UserForm(UserCreationForm):
    email = forms.EmailField(label="email")

    class Meta:
        model = User
        fields = ("username", "password1", "password2", "email")