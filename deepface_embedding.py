from deepface import DeepFace

source_img = "face/find_test1.jpg"
vector = DeepFace.represent(source_img)[0]["embedding"]
print('維度：', len(vector))
