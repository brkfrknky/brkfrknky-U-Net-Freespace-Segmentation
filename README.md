# brkfrknky/U-Net-Freespace-Segmentation
This project purpose is detect drivable areas for autonomous vehicles. I'm using python, opencv etc. technologies.

# Some result of the project

![image](https://user-images.githubusercontent.com/76915533/128061835-99ad8ed9-b356-4d05-b969-cc75d4d57415.png)


# Dataset

The dataset consists of 2 different data; CFC60 and CFC120. The only difference in the images is the camera angles.

| CFC60 | CFC120 |
| ------ | ------ |
| ![image](https://user-images.githubusercontent.com/76915533/129486236-2a8fee85-03fc-470b-98ac-acdd4e7ea2d6.png) | ![image](https://user-images.githubusercontent.com/76915533/129486241-bb11481f-4a4d-4eab-be08-2e32c28e0e48.png) |

#Preproccess

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
