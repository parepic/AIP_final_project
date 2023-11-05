# Student Workspace for CS4365 Applied Image Processing

## Set up

### Installation
```
git clone https://github.com/parepic/AIP_final_project.git
cd AIP_final_project
```
### Environment
The code can be run on Windows OS.
The environment can be simply set up by Anaconda:
```
conda env create -f environment.yml
conda activate face_morphing
```
### Download landmark detection models
Run the following command in the root directory:
```
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
```
This will download SuperGluePretrainedNetwork for custom feature extraction.

Go to https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2 and download shape_predictor_68_face_landmarks.dat.bz2 file. Extract the file on the root directory.


### Test images
Some test images are provided in data.zip file. Extract the file and put the data folder in the root directory. Images for face morphing are inside data/face and custom objects are inside data/custom folder.


## Run the code

Run this code from the terminal for face morphing with test images:
```
python morph.py --img1_name="face1.jpg" --img2_name="face2.jpg" 

```
And for custom objects, run:
```
python morph.py --img1_name="tower1.jpg" --img2_name="tower2.jpg" --custom
```
The output video is saved in the output directory. To test with your own images, add your images to either data/faces or data/custom folder depending on the type of the object. Then replace img1_name and img2_name arguments with the name of your files.


## Arguments

| Args | Description
| :--- | :----------
| --img1_name | Name of the input image 1 from the data directory.
| --img2_name | Name of the input image 2 from the data directory.
| --size | Size of the output video (defaults to 256).
| --custom | Use this flag if the objects are custom (non-face).


## Explanation
| Feature | Location in the code
| :--- | :----------
| Load 2 RGB images | morph.py: lines 297-300. 2 images are loaded and resized using opencv
| Run a pretrained landmark detector on images | UI.py: lines 394-398. An appropriate landmark detector is chosen and run based on if --custom flag is set.
| Interpolate the landmark positions and colors | morph.py: lines 326-329. First, the shepard_interpolate function is called for both images to create the mesh. Later, the applyMesh function applies the mesh on the input image. 
| Project the image to pre-trained GAN | Not implemented. The GPU of the DelftBlue supercomputer was being relied on during development of GAN inversion, and in the final stages of implementation, the jobs couldn’t run on the supercomputer for unknown reasons. Therefore, GAN inversion could not be fully integrated.
| Repeat the steps | morph.py: line 322. The process is repeated 20 times to create a morphing sequence.  
| Save the video | morph.py: line 338. The output video is saved in the ./output directory. 
| Automatically densify the landmarks | morph.py: line 307. Function densify_landmarks is called to densify the landmarks for both images using subdivision.
| Support for objects other than faces | UI.py: lines 394-398. The pretrained feature extractor for custom objects is executed if the --custom flag is set. This feature extractor works on any type of object but works best if the shapes of the images are very similar.




