from Data import *
import Draw
import random

def mod_relu(x, n=60):
  if x >= n: return x
  else: return 0

def averages(dataset, activation=mod_relu):
  '''
  Return the average images for each class.
  '''
  avgs = [[0] * (28*28) for _ in range(62)]
  counts = [0 for _ in range(62)]
  for label, image in zip(dataset.labels(), dataset.images()):
    counts[label] += 1
    avgs[label] = [img + avg for img, avg in zip(image, avgs[label])]
  avgs = [[a/count for a in avg] for avg, count in zip(avgs, counts)]
  avgs = [[activation(a) for a in avg] for avg in avgs]
  return avgs

if __name__ == '__main__':
  dataset = Data()
  print(len(dataset.images()))
  print(len(dataset.labels()))
  avgs = averages(dataset)
  i = random.randint(0,61)
  Draw.display_img(avgs[i])
