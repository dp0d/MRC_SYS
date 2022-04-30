from django import forms

from captcha.fields import CaptchaField

class UserForm(forms.Form):
    username = forms.CharField(max_length=128, widget=forms.TextInput(attrs={'class': 'form-control','style':'background-color: #20c997','autocomplete':"off"}))
    password = forms.CharField(max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control','style':'background-color: #20c997'}))

    captcha = CaptchaField(error_messages={"invalid":u"验证码错误","required":u"请输入验证码"})

class RegisterForm(forms.Form):
    gender = (
        ('male', "Boy"),
        ('female', "Girl"),
    )
    username = forms.CharField(label="用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control','style':'background-color: #20c997','autocomplete':"off"}))
    password1 = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control','style':'background-color: #20c997','autocomplete':"off"}))
    password2 = forms.CharField(label="确认密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control','style':'background-color: #20c997'}))
    email = forms.EmailField(label="邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control','style':'background-color: #20c997','autocomplete':"off"}))
    sex = forms.ChoiceField(label='性别', choices=gender)
    captcha = CaptchaField(error_messages={"invalid":u"验证码错误","required":u"请输入验证码"})
