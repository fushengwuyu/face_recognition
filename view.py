# author: sunshine
# datetime:2022/12/6 下午1:52
from sanic.blueprints import Blueprint
from sanic.views import HTTPMethodView
from sanic_openapi import doc
import urllib.request
import numpy as np
import cv2
from face_utils import FaceRecognitionOpenCV
from sanic.response import json
from config import args
import traceback

detect_model = 'models/face_detection_yunet_2022mar.onnx'
feature_model = 'models/face_recognition_sface_2021dec.onnx'

face_recognition = FaceRecognitionOpenCV(detect_model, feature_model, args.vector_path,
                                         input_size=(args.image_width, args.image_length))

face_bp = Blueprint("face")

from minio import Minio

client = Minio(f'{args.minio_ip}:{args.minio_port}',
               access_key=args.minio_access,
               secret_key=str(args.minio_secret),
               secure=False)


def fetch_image(image_url, timeout=10):
    """远程读取图片
    """
    try:
        resp = urllib.request.urlopen(image_url, timeout=timeout)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as err:
        return None


def read_remote_file(file_url):
    fields = file_url.lstrip('http://').split('/')
    bucket, file_path = fields[0], '/'.join(fields[1:])

    data = client.get_object(bucket, file_path)
    file_bytes = bytearray()
    for d in data.stream(32 * 1024):
        file_bytes += d

    # content = file_bytes.decode('utf-8')

    content = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    return content


def parse_request(request):
    if args.img_path_type == 0:
        # buffer
        img_origin = request.files.get('img')
        img = cv2.imdecode(np.frombuffer(img_origin.body, np.uint8), cv2.IMREAD_COLOR)
    elif args.img_path_type == 1:
        # http server
        img_url = request.json['img']
        img = fetch_image(img_url)
    elif args.img_path_type == 2:
        # minio
        img_url = request.json['img']
        img = read_remote_file(img_url)
    else:
        raise Exception("img_path_type not in [0, 1, 2]")
    return img


class FaceAdd(HTTPMethodView):

    @doc.consumes(doc.JsonBody({"img_url": doc.String("图片url地址，不能是本地图片")}), location="body")
    async def post(self, request):
        """
        人脸入库
        """

        try:
            img = parse_request(request)
            idx = face_recognition.put_face(img)
            return json({"code": 200, "message": "ok", "face_idx": idx})
        except Exception as e:
            print(traceback.print_exc())
            return json({"code": 300, "message": str(e)})


class FaceRetrieve(HTTPMethodView):
    @doc.consumes(doc.JsonBody({"img_url": doc.String("图片url地址，不能是本地图片")}), location="body")
    async def post(self, request):
        try:
            img = parse_request(request)
            _, item = face_recognition.face_embedding(img, args.threshold)

            return json({"code": 200, "message": "ok", "data": item})
        except Exception as e:
            return json({"code": 300, "message": str(e)})


class FaceSave(HTTPMethodView):
    async def post(self, request):
        """
        保存向量
        :param request:
        :return:
        """
        face_recognition.save()

        return json({"code": 200, "message": "ok"})


face_bp.add_route(FaceAdd.as_view(), '/face_add')
face_bp.add_route(FaceRetrieve.as_view(), '/face_retrieve')
face_bp.add_route(FaceSave.as_view(), '/face_save')
