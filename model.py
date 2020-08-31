from wtforms import Form, StringField, validators

class InputCommentForm(Form):
    r = StringField(validators=[validators.InputRequired()])
