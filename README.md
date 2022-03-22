# mosaic
Mosaic creator 

A simple application that builds a mosaic picture from smaller images (written in Go).
My original plan was to build an octree containing all the images in a given path, associated with their "average" color, but it turns out it's faster to just use random images and blend them a bit harder.

Command-line options:
