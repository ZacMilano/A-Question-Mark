# try edge detection 3x3 filter w/ -1's surrounding an 8

def convolve(img, filt):
	"""
	performs a convolution given an image and a filter, assuming the filter has dimensions <= the image in every dimension

	>>> convolve([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
	[[4, 3, 4], [2, 4, 3], [2, 3, 4]]

	"""
	img_height = len(img)
	img_width = len(img[0])
	filt_height = len(filt)
	filt_width = len(filt[0])

	res = []
	for _ in range(0, img_height - filt_height + 1):
		res.append([None] * (img_width - filt_width + 1))

	for y in range(0, img_height - filt_height + 1):
		for x in range(0, img_width - filt_width + 1):
			res[y][x] = equal_size_matrix_dot(get_sub_matrix(img, x, y, filt_width, filt_height), filt)

	return res


def pool(img, width, height):
	"""
	Downsample the image to a specified dimension.
	Assumes img_width % width == 0 and img_height % height == 0

	>>> pool([[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0]], 2, 2)
	[[1, 2], [3, 4]]

	>>> pool([[1, 0, 2, 0, 3, 4], [0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 1, 2], [0, 0, 4, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 7]], 2, 2)
	[[1, 2, 4], [3, 4, 2], [1, 1, 7]]

	"""
	res = []
	
	for y in range(0, round(len(img) / height)):
		res.append([None] * round(len(img[y]) / width))

	for y in range(0, round(len(img) / height)):
		for x in range(0, round(len(img[y]) / width)):
				res[y][x] = get_max_matrix(get_sub_matrix(img, x * width, y * height, width, height))

	return res




def get_sub_matrix(m, x, y, width, height):
	"""
	Takes a matrix, two coordinates, and two lengths.
	Assumes bounds are correct.

	Returns a submatrix with said variables of m.

	>>> get_sub_matrix([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]], 1, 1, 3, 3)
	[[1, 1, 1], [0, 1, 1], [0, 1, 1]]

	"""
	res = []
	for i in range(y, y + height):
		res.append(m[i][x:x + width])
	return res


def get_max_matrix(m):
	"""
	Returns the maximum element from the matrix (2 D).
	"""
	res = m[0][0]
	for row in m:
		for elem in row:
			res = max(res, elem)
	return res
	

def equal_size_matrix_dot(m1, m2):
	"""
	Assumes the sizes of m1 and m2 are equal.

	Computes the dot product of the flattened matrices.

	>>> equal_size_matrix_dot([[1, 1, 1], [0, 1, 1], [0, 0, 1]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
	4

	"""
	res = 0

	for y in range(0, len(m1)):
		for x in range(0, len(m1[y])):
				res += m1[y][x] * m2[y][x]

	return res


if (__name__ == '__main__'):
	import doctest
	doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)
