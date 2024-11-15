## Simple Head Pose Annotation Tool
### install method
pip install -r requirements.txt <br>
python ann3d.py <br>
**Note: This has only been tested on Windows 10 with Python version 3.12.7.**
### 使用方法
![alt text](image.png)
Select a folder containing .jpeg, .jpg, and .png files (other image types are currently not supported). Currently, all images must be head shots. Please crop the head shots as shown below:<br>
![alt text](5.jpg)<br>
After selecting the folder, the software interface will look like this, with some operations also shown in the image below:<br>
![alt text](1731380850191.png)<br>
Pressing the 'A' key on the keyboard shows the previous image, while pressing the 'D' key shows the next image. In the operation interface, holding down the left mouse button and dragging can change the model's orientation, with a preview of the orientation displayed on the image. In the operation interface, there is a circle; dragging the mouse inside the circle performs X/Y-axis rotation, while dragging outside the circle performs Z-axis rotation. Pressing the right mouse button will reset the model. The 'Delete' key on the keyboard can delete the annotation for the current image and clear the record in the JSON file.<br>
The contents of the folder after annotation are shown below:<br>
![alt text](image-1.png)<br>
The folder will contain additional JSON files that can be used to view the output Euler angles and a 4x4 matrix, as shown below:<br>
![alt text](1731381702811.png)
