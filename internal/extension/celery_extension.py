from celery import Task, Celery
from flask import Flask


def init_app(app: Flask):
    class FlaskTask(Task):
        """定义FlaskTask"""

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()

    app.extensions["celery"] = celery_app
