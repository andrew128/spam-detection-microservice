from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class CommentForm(FlaskForm):
    comment = StringField('comment', validators=[DataRequired()])

class NewLFForm(FlaskForm):
    word = StringField('word', validators=[DataRequired()])