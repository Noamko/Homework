import matplotlib.pyplot as plt
import numpy as np
import sys

def kmeans(pixels, cents):
    def pixelDistance(p1, p2):
        return np.linalg.norm(p2-p1)

    # initialization
    pixels = pixels.astype(float) / 255.0
    pixels = pixels.reshape(-1, 3)

    # loop through pixels
    previousCenteroids = None
    closestCenteroid = None
    result = ""
    for i in range(20):
        pixelClusterMean = {}; groupsizes = {}; res = []; means = []
        for pixel in pixels:
            temp = 999

            # find the closest centeroid to our pixel
            for centeroid in cents:
                dist = pixelDistance(pixel, centeroid)
                if temp > dist:
                    temp = dist
                    closestCenteroid = centeroid
            ###############################################

            ## we sum the pixel RGB to the closest centeroid cluster
            key = str(closestCenteroid)
            if key in pixelClusterMean: 
                pixelClusterMean[key][0] += pixel[0]
                pixelClusterMean[key][1] += pixel[1]
                pixelClusterMean[key][2] += pixel[2]
                groupsizes[key] += 1
            else:
                pixelClusterMean[key] = [pixel[0],pixel[1],pixel[2]]
                groupsizes[key] = 1
            res.append(closestCenteroid)
        for ceteroid in cents:
            key = str(ceteroid)
            numofpixels = groupsizes[key]
            pixelClusterMean[key][0] /= numofpixels
            pixelClusterMean[key][1] /= numofpixels
            pixelClusterMean[key][2] /= numofpixels

            pixelClusterMean[key][0] = pixelClusterMean[key][0].round(4)
            pixelClusterMean[key][1] = pixelClusterMean[key][1].round(4)
            pixelClusterMean[key][2] = pixelClusterMean[key][2].round(4)

            means.append(pixelClusterMean[key])
        if means == previousCenteroids: break
        else:previousCenteroids = means
        cents = means
        result += f"[iter {i}]:{','.join([str(i) for i in means])}\n"
    return result


image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)
orig_pixels = plt.imread(image_fname)

km = kmeans(orig_pixels, z)
outfile = open(out_fname,"w")
outfile.write(km)