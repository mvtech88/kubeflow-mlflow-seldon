from io import BytesIO
import PIL.Image as Image
import base64
import requests
import sys
import logging

logger = logging.getLogger('__mymodelclient__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    
    url = sys.argv[1]
    path = sys.argv[2]
    img = Image.open(path)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    #print(im_b64)
    im_b64 = im_b64.decode()
    #print(im_b64)
    
    #data = {}
    data = {'data': {'names': [], 'ndarray': [im_b64]}}

    logger.info("sending image {} to {}".format(path, url))
    
    response = requests.post(url, json = data, timeout = None)
    logger.info("caught response {}".format(response))
    status_code = response.status_code  
    result = response.json()
    
    print(result)
