from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()

class AddRecordForm(forms.Form):
    month = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={"placeholder": "YYYY-MM", "style": "width:140px"})
    )
    demand = forms.FloatField(
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 1500", "style": "width:140px", "step": "any"})
    )
