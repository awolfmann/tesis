import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from os.path import join, exists, basename, splitext


def plot_scores(path, image_name):
    csv_file = join(path,'scores.csv')
    dataArray = np.genfromtxt(csv_file, delimiter = ',', names = True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_title("Scores-Style")    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Score')
    for col_name in dataArray.dtype.names:
        if col_name == 'iteration':
            x = dataArray[col_name]
        else:
            ax1.plot(x, dataArray[col_name], label=col_name)

    leg = ax1.legend(loc=7, ncol=2)

    offset = .075
    for i, it in enumerate(range(0, 1001, 200)):
        image_name_in = image_name +'_{:d}.png'.format(it)
        image_file = join(path, image_name_in)
        a1 = plt.axes([offset, .9, .1, .1])
        plt.setp(a1, xticks=[], yticks=[])
        img=mpimg.imread(image_file)
        plt.imshow(img)

        offset += .155

    result_file = join(path, 'scores.png')
    plt.savefig(result_file)

# plot_scores(path='/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/',
#     image_name='pitt_matisse')

# # the main axes is subplot(111) by default
# plt.plot(t, s)
# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
# plt.xlabel('time (s)')
# plt.ylabel('current (nA)')
# plt.title('Gaussian colored noise')

# # this is an inset axes over the main axes
# a1 = plt.axes([.075, .9, .1, .1])
# plt.setp(a1, xticks=[], yticks=[])
# image_file = '/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/pitt_matisse.png'
# img=mpimg.imread(image_file)
# plt.imshow(img)

# a = plt.axes([.23, 0, .095, .095])
# plt.setp(a, xticks=[], yticks=[])
# image_file = '/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/pitt_matisse.png'
# img=mpimg.imread(image_file)
# plt.imshow(img)

# b = plt.axes([.385, 0, .1, .1])
# plt.setp(b, xticks=[], yticks=[])
# image_file = '/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/pitt_matisse.png'
# img=mpimg.imread(image_file)
# plt.imshow(img)

# c = plt.axes([.54, 0, .1, .1])
# plt.setp(c, xticks=[], yticks=[])
# image_file = '/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/pitt_matisse.png'
# img=mpimg.imread(image_file)
# plt.imshow(img)

# d = plt.axes([.695, 0, .1, .1])
# plt.setp(d, xticks=[], yticks=[])
# image_file = '/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/pitt_matisse.png'
# img=mpimg.imread(image_file)
# plt.imshow(img)

# e = plt.axes([.85, 0, .1, .1])
# plt.setp(e, xticks=[], yticks=[])
# image_file = '/home/ariel/Documents/tesis/neural-style/tests/pitt_matisse/nin_10_1000_image/pitt_matisse.png'
# img=mpimg.imread(image_file)
# plt.imshow(img)

# plt.savefig('foo.png')
# n, bins, patches = plt.hist(s, 400, normed=1)
# plt.title('Probability')
# plt.xticks([])
# plt.yticks([])

# plt.savefig('foo.png', bbox_inches='tight')

# plt.show()

# # this is another inset axes over the main axes
# a = plt.axes([0.2, 0.6, .2, .2], axisbg='y')
# plt.plot(t[:len(r)], r)
# plt.title('Impulse response')
# plt.xlim(0, 0.2)
# plt.xticks([])
# plt.yticks([])

# plt.show()
# os.mkdir