# author: sunshine
# datetime:2022/12/6 下午2:27
from sanic import Sanic
from sanic_cors import CORS
from sanic_openapi import openapi2_blueprint
from view import face_bp, face_recognition
from sanic.blueprints import Blueprint
from config import args


def create_app():
    """初始化app
    """
    app = Sanic(__name__)

    # 跨域
    CORS(app)

    # 添加blueprint
    component_bp = Blueprint.group(
        face_bp,
        openapi2_blueprint,
        url_prefix='/face'
    )
    # Blueprint.
    app.blueprint(component_bp)
    return app


app = create_app()


@app.before_server_stop
async def save_vector(app, loop):
    face_recognition.save()


if __name__ == '__main__':
    app.run('0.0.0.0', port=args.port)
