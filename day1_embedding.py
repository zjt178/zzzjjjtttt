# import numpy as np#导入numpy库
#
# word_to_idx = {"猫": 0, "狗": 1, "汽车": 2}#定义一组字典
#
# embedding_table = np.array([
#     [0.1, 0.8, 0.3],
#     [0.2, 0.7, 0.4],
#     [0.9, 0.1, 0.2]
# ])#定义一个词向量矩阵
#
# cat_vec = embedding_table[word_to_idx["猫"]]
# #获取猫的词向量，word_to_idx["猫"]得到字典中的值为0，embedding_table得到列表中的第一个元素
# print("猫的向量:", cat_vec)
#
# dog_vec = embedding_table[word_to_idx["狗"]]
# #获取狗的词向量，word_to_idx["狗"]得到字典中的值为1，embedding_table得到列表中的第二个元素
# print("狗的向量:", dog_vec)
#
# similarity = np.dot(cat_vec, dog_vec) / (np.linalg.norm(cat_vec) * np.linalg.norm(dog_vec))
# #计算猫和狗的相似度（用余弦来表示）np.dot：向量的点积，np.linalg.norm：求向量的模
# print(f"猫和狗的相似度: {similarity:.4f}") #保留四位小数
#
# car_vec = embedding_table[word_to_idx["汽车"]]
# similarity2 = np.dot(cat_vec, car_vec) / (np.linalg.norm(cat_vec) * np.linalg.norm(car_vec))
# #计算猫和汽车的相似度用余弦来表示）np.dot：向量的点积，np.linalg.norm：求向量的模
# print(f"猫和汽车的相似度: {similarity2:.4f}") #保留四位小数


#检验
import numpy as np
word_to_idx = {"人": 1, "鸟": 2, "鱼": 3}
embedding_table = np.array([
    [0.1, 0.8, 0.3],
    [0.2, 0.7, 0.4],
    [0.9, 0.1, 0.2],
    [0.4, 0.5, 0.6]
])
person_vec = embedding_table[word_to_idx["人"]]
print(f"人的词向量{person_vec}")

fish_vec = embedding_table[word_to_idx["鱼"]]
print(f"鱼的词向量{fish_vec}")

cos_sim = np.dot(person_vec, fish_vec) / (np.linalg.norm(person_vec) * np.linalg.norm(fish_vec))
print(f"人和鱼的相似度{cos_sim:.4f}")
bird_vec = embedding_table[word_to_idx["鸟"]]

cos_sim2 = np.dot(person_vec, bird_vec) / (np.linalg.norm(person_vec) * np.linalg.norm(bird_vec))
print(f"人和鸟的相似度{cos_sim2:.4f}")
