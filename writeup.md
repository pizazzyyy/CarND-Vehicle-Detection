
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation. The project is based on the work from this [git repo](https://github.com/dolaameng/CarND-Vehicle-Detection) and I make some modifications to make the detection more robust. Details will be described below.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I am using skimage.feature.hog to extract hog features from the vehicle/non-vehicle images. The HOG features are from RGB based gray images. The related code are implemented in sdcvehicle.classification.ImageFeatExtractor class. 

I started by reading in all the `vehicle` and `non-vehicle` images. Here is examples using the gray image and HOG parameters of `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG output][https://github.com/yyporsche/CarND-Vehicle-Detection/blob/master/output_images/HOG.png]

####2. Explain how you settled on your final choice of HOG parameters.
Hyperparameters, such as pixels_per_cell, cells_per_block of HOG, hist_nbins from color-histogram features, and C from SVC, are tuned by a RandomizedSearch based on cross validation (from sklearn.model_selection.RandomizedSearchCV. After the optimal values are estimated, they are fixed and used across the whole training process. This is implemented in sdcvehicle.classification.fit_best_model().
The classifier is implemented in sdcvehicle.classification.VehicleClassifier class. It consists of two steps
        feature_extractor: it is a combination of HOG features and color HISTOGRAMS., which are implemented in sdcvehicle.classification.ImageFeatExtractor class.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a standard LinearSVC from sklearn package. There are two parts of training data: original "vehicle" vs "nonvehicle" from Udacity. To further reduce false positives, the data is augmented:
1. augmented negative examples: random sampling of patches from non-vehicle areas, it is implemented in sdcvehicle.classification.enhance_negative_data(), in order to get more robustness, I added more pictures to the original work. This will prevent overfitting.
2. augmented positive examples: random shifting, rotation, and cropping of original vehicle images

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code of detection is implemented in sdcvehicle.detection.VehicleDetector class. I have tried two ways of sampling patches from images.
Random sampling as in VehicleDetector.get_random_patches() method. I found it can give good results but it is more expensive to get comparative results.
Sliding window + Image Pyramid as in VEhicleDetector.get_pyramid_slide_window(): it iterates through a pyramid of scaled images, so that vehicles with different sizes can be detected with the trained classifier on fixed size images, and then iterates through a sliding with with overlaps, so that images on different locations can be detected. The parameters such as sliding window overlap, scaling factor, and start/end space to search, are determined by visual inspection of intermediate results and trial-and-error.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Vehicle detection in images is implemented in sdcvehicle.detection.VehicleDetector.detect_in_image() method. The main steps are,
1. sample different patches from image by a sliding window on a pyramid of scaled images.
2. predict vehicle/non-vehicle for each patch
3. construct a heatmap (based on sum) of predictoin probability based on predictions on each patch
4. merge the patch to get bounding boxes from heatmap.

Some results are shown here. I find that change the heat map threshold from original 2.5 to 4 or even 5 will eliminate some false positive and also improve the performance:

![Heatmap threshold 4][https://github.com/yyporsche/CarND-Vehicle-Detection/blob/master/output_images/output_heatmap_4.png]
![Heatmap threshold 5][https://github.com/yyporsche/CarND-Vehicle-Detection/blob/master/output_images/output_heatmap_5.png]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As described above, the boxes close enough to each other are merged by following steps (implemented in sdcvehicle.detection.VehicleDetector.draw_merged_boxes()

1. generating heatmap based on sum of individual bounding boxes
2. thresholding heatmap to generate binary images
3. morphology operation such as opening to reduce noise
4. label the image by skimage.measure.label
5. find bounding box of each labeled region as detections.

I also added a time based decay system for heatmap is to improve the performance (as in detect_in_video method), this will track the last 3 frames heat map and have corresponding weights on current frame.

```
def process_frame(frame):
            if self.furthest_heatmap is None:
                self.furthest_heatmap = self.get_heatmap(frame)
                detection_img = self.draw_merged_boxes(frame, self.furthest_heatmap)            
            elif self.last_last_heatmap is None:
                self.last_last_heatmap = self.get_heatmap(frame)
                detection_img = self.draw_merged_boxes(frame, self.last_last_heatmap)
            elif self.last_heatmap is None:
                self.last_heatmap = self.get_heatmap(frame)
                detection_img = self.draw_merged_boxes(frame, self.last_heatmap)
            else:
                heatmap = self.get_heatmap(frame)
                combined_heatmap = heatmap * 0.5 + self.last_heatmap * 0.2 + self.last_last_heatmap * 0.2 + self.furthest_heatmap * 0.1
                detection_img = self.draw_merged_boxes(frame, combined_heatmap)
                self.furthest_heatmap = self.last_last_heatmap
                self.last_last_heatmap = self.last_heatmap
                self.last_heatmap = heatmap
            return detection_img
```

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The final results still have many false positives after all the tuning and efforts. I will spend more time to optimize the sliding window and also think about adding features into the model.
