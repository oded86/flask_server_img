import requests
from PIL import Image


url = 'http://127.0.0.1:8020/runClassify'
image = Image.open("poopTest2.jpeg")
width, height = image.size
img = open('poopTest2.jpeg', 'rb')
my_img = {'image': img}
r = requests.post(url, files=my_img)

# convert server response into JSON format.
print(r.json()['msg'])