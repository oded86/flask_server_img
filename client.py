import requests
from PIL import Image


url = 'http://127.0.0.1:8020/poopOrNot'
image = Image.open("11 (5).jpeg")
width, height = image.size
img = open('11 (5).jpeg', 'rb')
my_img = {'image': img}
r = requests.post(url, files=my_img)

# convert server response into JSON format.
print(r.json()['msg'])