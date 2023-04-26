# author: sunshine
# datetime:2023/4/14 下午3:58
import argparse
from license_utils import LicenseDecode

parser = argparse.ArgumentParser(
    description='A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--vector_path', type=str, default='./vector.bin', help='Path to the vector')
parser.add_argument('--port', type=int, default=5003, help='server port')
parser.add_argument('--image_width', type=int, default=480, help='image_width')
parser.add_argument('--image_length', type=int, default=640, help='image_length')
parser.add_argument('--threshold', type=float, default=0.363, help='cosin threshold')
parser.add_argument('--score_threshold', type=float, default=0.9, help='score_threshold')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='nms_threshold')
parser.add_argument('--max_elements', type=int, default=20000, help='max_elements')
parser.add_argument('--minio_ip', type=str, default='192.168.0.79', help='minio_ip')
parser.add_argument('--minio_port', type=int, default=9000, help='minio_port')
parser.add_argument('--minio_access', type=str, default='minio', help='minio_access')
parser.add_argument('--minio_secret', type=str, default='12345678', help='minio_secret')
parser.add_argument('--license_path', type=str, default='./License.dat', help='license_path')
parser.add_argument('--img_path_type', type=int, default=2, help='0: buffer; 1: http server; 2: minio')
args = parser.parse_args()

LicenseDecode(args.license_path).license_check()