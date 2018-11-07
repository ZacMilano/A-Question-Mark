from Data import Data


def add_images(img1, img2):
    res = []

    for i in range(0, len(img1)):
        res.append(img1[i] + img2[i])

    return res

def map_avg_img(img, n):
    for i in range(0, len(img)):
        img[i] /= n

    return img

def dot_imgs(img1, img2):
    res = 0

    for i in range(0, len(img1)):
        res += img1[i] * img2[i]

    return res

class Similarity:
    class ImageTrainingTup:
        def __init__(self, label, image_sum, image_count):
            self.label = label
            self.image_sum = image_sum
            self.image_count = image_count

    IMG_HEIGHT = 28
    IMG_WIDTH = 28

    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.training_labels = self.training_data.labels()
        self.training_images = self.training_data.images()
        self.test_data = test_data
        self.test_labels = self.test_data.labels()
        self.test_images = self.test_data.images()

    def average_training_images(self):
        training_image_dict = {}

        for label in self.training_labels:
            null_img = [0]*(self.IMG_WIDTH * self.IMG_HEIGHT)
            training_image_dict[label] = self.ImageTrainingTup(label, null_img, 0)

        for i in range(0, len(self.training_images)):
            cur_img = self.training_images[i]
            cur_label = self.training_labels[i]
            prev_img_tup = training_image_dict[cur_label]
            sum_imgs = add_images(prev_img_tup.image_sum, cur_img)
            training_image_dict[prev_img_tup.label] = self.ImageTrainingTup(prev_img_tup.label, sum_imgs, prev_img_tup.image_count + 1)

        res_dict = {}
        for label in training_image_dict:
            img_tup = training_image_dict[label]
            res_dict[img_tup.label] = map_avg_img(img_tup.image_sum, img_tup.image_count)

        return res_dict

    def guess_img(self, index, training_dictionary):
        real_label = self.test_labels[index]
        real_img = self.test_images[index]

        max_seen = (-1, -1)
        for label in training_dictionary:
            dot_val = dot_imgs(training_dictionary[label], real_img)
            if dot_val > max_seen[1]:
                max_seen = (label, dot_val)

        print('real label: ' + str(real_label) + ', guessed label: ' + str(max_seen[0]))
        return max_seen[0]



if __name__ == "__main__":
    d = Data()
    s = Similarity(d, d)
    t_d = s.average_training_images()
    for i in range(0, 10):
        s.guess_img(i, t_d)
