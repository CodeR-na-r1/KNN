import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm import tqdm
from skimage.measure import regionprops, label

text_images = [plt.imread(path) for path in pathlib.Path("out/").glob("*.png")]
print(f"Total images: {len(text_images)}")

train_images = {}
for path in tqdm(sorted(pathlib.Path("out/train").glob("*"))):
    symbol = path.name[-1]
    train_images[path.name] = []
    for image_path in sorted(path.glob("*.png")):
        train_images[path.name].append(plt.imread(image_path))

def extract_features(image):
    if image.ndim == 3:
        gray = np.mean(image, 2)
        gray[gray > 0] = 1
        labeled = label(gray)
    else:
        labeled = image.astype("uint8")
    props = regionprops(labeled)[0]
    extent = props.extent
    eccentricity = props.eccentricity
    euler = props.euler_number
    rr, cc = props.centroid_local
    rr = rr / props.image.shape[0]
    cc = cc / props.image.shape[1]
    feret = (props.feret_diameter_max - 1) / np.max(props.image.shape)

    return np.array([extent, eccentricity, euler, rr, cc, feret], dtype="f4")

knn = cv.ml.KNearest_create()

train = []
responses = []

sym2class = {symbol: i for i, symbol in enumerate(train_images)}
class2sym = {value: key for key, value in sym2class.items()}

for _, symbol in tqdm(enumerate(train_images)):
    for image in train_images[symbol]:
        train.append(extract_features(image))
        responses.append(sym2class[symbol])

train = np.array(train, dtype="f4")
responses = np.array(responses).reshape(-1, 1).astype("f4")

knn.train(train, cv.ml.ROW_SAMPLE, responses)

# features = extract_features(train_images["A"][0]).reshape(1, -1)

# res, results, neigbours, dist = knn.findNearest(features, 5)
# print(class2sym[int(res)])

def image2text(image) -> str:
    gray = np.mean(image, 2)
    gray[gray > 0] = 1
    labeled = label(gray)
    regions = regionprops(labeled)

    answer = []

    sorted_regions = sorted(regions, key=lambda it: it.centroid[1])     # regions, отсортированные по centroid.x (чтобы буквы распознавались по порядку)
    regions_x = [region.centroid[1] for region in sorted_regions]   # содержат только centroid.x
    skip = False     # пропуск второй части буквы i (либо точка либо палка)

    max_space_distance = 0   # максимальное расстояние между словами для данной фразы
    for i in range(0, len(sorted_regions) - 1):
        temp_distance = ((sorted_regions[i + 1].centroid[1] - sorted_regions[i + 1].image.shape[1] / 2) - # левый край правой буквы - правый край левой быквы
                    (sorted_regions[i].centroid[1] + sorted_regions[i].image.shape[1] / 2))
        if temp_distance > max_space_distance:
             max_space_distance = temp_distance

    for i, region in enumerate(sorted_regions):
        if skip:     # пропуск второй части буквы i (либо точка либо палка)
            skip = False
            continue

        if (i > 0 and
            abs(((sorted_regions[i].centroid[1] - sorted_regions[i].image.shape[1] / 2) -
                          (sorted_regions[i - 1].centroid[1] + sorted_regions[i - 1].image.shape[1] / 2)) - max_space_distance) < 20):     # если расстояние близко к максимальному, добавляем пробел
            answer.append(" ")

        if i < (len(regions_x) - 1) and regions_x[i + 1] - regions_x[i] < 15:     # если второй регион слишком близко, то это часть буквы i
            answer.append("i")
            skip = True
            continue

        features = extract_features(region.image).reshape(1, -1)
        ret, results, neigbours, dist = knn.findNearest(features, 5)
        answer.append(class2sym[int(ret)][-1])

    return "".join(answer)

print("\n==== RESULTS ====")

for i, image in enumerate(text_images):
    print(f"Text from image №{i+1} -> {image2text(image)}")
print("==========\n")