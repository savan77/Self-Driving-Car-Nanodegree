# **Finding Lane Lines on the Road** 



[image1]: ./test_images_output/solidWhiteRight.jpg "Lane Detection"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline to detect lanes consists of 6 tasks:
* Convert given frame to grayscale
* Apply Gaussian Blur in order to use Canny Edge Detection
* Apply Canny Edge Detection
* Find Region of Interest (RoI) through trial and error process
* Use Hough Transform to find line segments
* Draw detected lanes on original frame

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function to accept one more parameter.  When this parameter(`extrapolate`) is `True`, `draw_lines()` function will call another function which finds the single line from line segments. I used the hint provided in Jupyter notebook to calculate this line. 
- First, I calculated average slope, average x, and average y from given line segments. 
- But before that I need to separate out left and right lanes. In order to do so, I used slope of given line segment to find out whether it is left or right lane.
- Then, I calculated extreme points (top/bottom) by assuming fixed position of corresponding y.


### 2. Identify potential shortcomings with your current pipeline

I ran my pipeline on `challenge.mp4` and identified some problems.
* Hough Transform require scene specific parameter tuning. Even though the parameters I found were working well on other two videos, it didn't gave good result on challenge.mp4 because there were other lines that were detected by Hough Transform.
* I also noticed algorithm didn't work when there was a different road color or change in light/shadow.
* Since my current implementation uses slope to find full line, sometimes it makes mistakes when there isn't proper curve on road.



### 3. Suggest possible improvements to your pipeline

* I guess Hough Transform would not work when there is sudden change in lighting/shadow, or lane curves. I think color based detection needs to be used with some modifications to make sure it works with common lane colors.
