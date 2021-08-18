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
       one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
       for i,unique_value in enumerate(np.unique(res_mask)):
           one_hot[:,:,i][res_mask==unique_value]=1
       return one_hot
 ```
 
 You can find the codes of this section from the preprocess.py file.
 
 
 # Model

U-Net is more successful than conventional models, in terms of architecture and in terms pixel-based image segmentation formed from convolutional neural network layers. It’s even effective with limited dataset images. The presentation of this architecture was first realized through the analysis of biomedical images. For this reason, U-Net architecture was used in the project.

![image](![u-1](https://user-images.githubusercontent.com/76915533/129893959-bf9733c7-db7d-404d-9eea-16bd2a2362e3.jpg))


# Augmentation

Data augmentation is used to increase the accuracy of the model and to perform better in some rare cases.

```sh
for image in tqdm.tqdm(train_input_path_list):
    img=Image.open(image)
    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
    img_aug = color_aug(img)
    new_path=image[:-4]+"-1"+".png"
    new_path=new_path.replace('images', 'augmentation')
    img_aug=np.array(img_aug)
    cv2.imwrite(new_path,img_aug)
 ```
 
 | Original | Augmentation |
| ------ | ------ |
| ![image](https://user-images.githubusercontent.com/76915533/129894490-f754fdef-10c4-48b6-b834-3395a97d6325.png) | ![image](https://user-images.githubusercontent.com/76915533/129894509-fc05e168-75e9-4d80-9952-2cef637518c5.png) |
