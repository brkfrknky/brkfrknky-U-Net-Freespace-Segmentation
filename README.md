# brkfrknky/U-Net-Freespace-Segmentation
This project purpose is detect drivable areas for autonomous vehicles. I'm using python, opencv etc. technologies.

# Some result of the project

![image](https://user-images.githubusercontent.com/76915533/128061835-99ad8ed9-b356-4d05-b969-cc75d4d57415.png)


# Dataset

The dataset consists of 2 different data; CFC60 and CFC120. The only difference in the images is the camera angles.

| CFC60 | CFC120 |
| ------ | ------ |
| ![image](https://user-images.githubusercontent.com/76915533/129486236-2a8fee85-03fc-470b-98ac-acdd4e7ea2d6.png) | ![image](https://user-images.githubusercontent.com/76915533/129486241-bb11481f-4a4d-4eab-be08-2e32c28e0e48.png) |

#Json Mask

Each original image has its own label json file. These json files hold the mask information of the image. The contents of this file are shown below.

```sh
{
            "id": 121522,
            "classId": 38,
            "description": "",
            "geometryType": "polygon",
            "labelerLogin": "abilgin4",
            "createdAt": "2020-07-02T06:19:05.765Z",
            "updatedAt": "2020-07-29T23:20:08.865Z",
            "tags": [],
            "classTitle": "Freespace",
            "points": {
                "exterior": [
                    [
                        0,
                        1208
                    ],
                    [
                        0,
                        836
                    ],
                    [
                        324,
                        751
                    ],
                    [
                        622,
                        674
                    ],
                    [
                        801,
                        630
                    ],
                    [
                        880,
                        611
                    ],
                    [
                        937,
                        599
                    ],
                    [
                        974,
                        592
                    ],                       
                ],
                "interior": []
            }
        }
```

Opencv's fillPoly function is used to use this information.

```sh
cv2.fillPoly(mask,np.array([obj['points']['exterior']]),color=1)
```
You can find the codes of this section from the json_mask.py file.

# Image Mask

This section exists only to make sure that the mask is applied correctly. Just check a few sample images while running the code.

You can find the codes of this section from the mask_image.py file.

#Preprocessing

In order for the data to be given to the model, it must be converted to tensor format. So reshaping and normalization is done on images and masks.

Normalization
```sh
img=cv2.imread(image) 
norm_img = np.zeros((1920,1208))
final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
```
Reshape
```sh
output_shape = (224, 224)
img=cv2.resize(final_img,tuple(output_shape))
```

One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. There are 2 categories in this project; freespace and background. 

```sh
def one_hot_encoder(res_mask,n_classes):
       #one hot encode
       #Create an np.array of zeros.
       one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
       #Find unique values in res_mask [0,1]
       #increase in i by the length of the list
       #[0,1] when returning the inside of list, each list element is given to unique_value variable
       for i,unique_value in enumerate(np.unique(res_mask)):
           one_hot[:,:,i][res_mask==unique_value]=1
       return one_hot
 ```
