y = roundit(np.pi, rmode=4, flip=0)
y == 3

A = np.array([[0, 1.1, 1.5], [1.9, 2.4, 0.5]])
B = roundit(A, flip=0, rmode=1)
np.array_equal(B, np.array([[0,1,2], [2,2,0]]))

B = roundit(A.T, flip=0, rmode=1)
np.array_equal(B, np.array([[0,1,2], [2,2,0]]).T)

A = np.array([[0, -1.1, -1.5], [-2.9, -2, -0.5], [0.5, 1.5, 3]])
B = roundit(A, flip=0, rmode=4);
np.array_equal(B, np.array([[0, -1, -1], [-2, -2, 0], [0, 1, 3]]))
