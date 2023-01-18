import skimage as sk


def is_artificial_circle(contour, shape, draw=False):
    """
    Check if the current contour is a artificial circle (dermatological microscope objective) or natural shape
    Using only Hough transform
        Based on parameters: n of edge points, accum sum of Hough 
        detects if the given edge is an artificial artefact (lense ocular) or natural skin feature
    """
    detected = False
    mask = np.zeros(shape)
    # mask = np.zeros(shape, dtype="uint8")
    cv2.drawContours(mask, contour,-1,1,2)

    # Hough circle detection with scikit-image
    # hough_radii = np.arange(330, 332, 1)
    hough_radii = [330]
    hough_res = sk.transform.hough_circle(mask, hough_radii)

    # Select the most prominent 1 circle
    accums, cx, cy, radii = sk.transform.hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1, normalize=False)
    non_zero = np.count_nonzero(mask)
    print(f"nonzero: {non_zero}")
    print(f"accums: {accums}")
    print("accums/non_zero: %.3f perc., radius: %d" %(accums[0]/non_zero * 100, radii[0] ))
    if accums[0]/non_zero > 0.005/100 and non_zero > 100:
        print(">>>>>> Occular detected <<<<<<<")
        detected = True

    # Draw circle
    if draw:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = sk.color.gray2rgb(mask)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = sk.draw.circle_perimeter(center_y, center_x, radius,
                                            shape=image.shape)
            image[circy, circx] = (220, 20, 20)
        image = np.clip(image, 0, 1)
        ax.imshow(image, cmap=plt.cm.gray)
        plt.show()
    return detected
