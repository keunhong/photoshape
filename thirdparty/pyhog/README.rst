
pyhog
----------

The Pascal VOC Toolkit comes with a Matlab/C implementation of HOG features by
Pedro Felzenszwalb, Deva Ramanan and presumably others. Since I'm not very fond
of Matlab I replaced the Matlab-specific parts for their Numpy equivalents. It
works, but it's not very efficient because it copies the array into a
Fortran-ordered version. That should be easy to fix.

See an example of here: http://nbviewer.ipython.org/github/dimatura/pyhog/blob/master/pyhog_example.ipynb

Daniel Maturana - dimatura@cmu.edu
2012
