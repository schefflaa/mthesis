# FlattendLR
This model is a simple linear regression model that uses following image:


|  Input Image  | 
|     :---:     |
| <img src="./tblogs/FlattendLR/viz/input(original).png" style="width: 25%;"/> |
| 187500(+1) features |

- *flattened (to a 1D vector)* as its input. 
- It cannot leverage the temporal information in the time series data as it only uses a single image as input. 

So for an $250 \times 250 \times 3$ image, the input is a $187500$ vector.
$250 \times 250 \times 3 = 187500$

<br/>
<br/>
<br/>

# FlattendBrightnessLR
This model is a simple linear regression model that uses following image (grayscale):

|  Input Image  | 
|     :---:     |
| <img src="./tblogs/FlattendBrightnessLR/viz/grayFulloriginal.png" style="width: 25%;"/> |
| 62500(+1) features |

- *flattened (to a 1D vector)* as its input. 
- It cannot leverage the temporal information in the time series data as it only uses a single image as input. 

Color channels are reduced to brightness → `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]`. 

So for an $250 \times 250$ image, the input is a $62500$ vector.
$250 \times 250 = 62500$

<br/>
<br/>
<br/>

# PatchMeanLR
This model is a simple linear regression model that uses the patch-means as input:

|  Patchsize 5  |  Patchsize 25  |  Patchsize 50  |
|     :---:     |     :---:      |      :---:     |
| <img src="./tblogs/PatchMeanLR/viz/5patch_means.png"/> | <img src="./tblogs/PatchMeanLR/viz/25patch_means.png"/> | <img src="./tblogs/PatchMeanLR/viz/50patch_means.png"/>|
| 7500(+1) features |  300(+1) features  |  75(+1) features   |

- Color channels are preserved, but *flattened (to a 1D vector)* as input. 
- It cannot leverage the temporal information in the time series data as it only uses a single image as input.

It divides the original $250 \times 250 \times 3$ image into patches of `patchsize` and calculates the mean of each patch for each color channel. 

E.g. for patchsize $50$, the input to the linear layer is a $75$ dimensional vector:
$\frac{250}{50} \times \frac{250}{50} \times 3 = 75$

<br/>
<br/>
<br/>

# PatchBrightnessLR
This model is a simple linear regression model that uses greyscale patches as input.

|  Patchsize 5  |  Patchsize 25  |  Patchsize 50  |
|     :---:     |     :---:      |      :---:     |
| <img src="./tblogs/PatchBrightnessLR/viz/gray5patch_means.png"/> | <img src="./tblogs/PatchBrightnessLR/viz/gray25patch_means.png"/> | <img src="./tblogs/PatchBrightnessLR/viz/gray50patch_means.png"/>|
| 2500(+1) features |  100(+1) features  |  25(+1) features   |

- Color channels are not preserved. Only the brightness channel is used. But *flattened (to a 1D vector)* as its input. 
- It cannot leverage the temporal information in the time series data as it only uses a single image as input.

It divides the original $250 \times 250 \times 3$ image into patches and calculates the mean brightness of each patch (color channels are reduced to brightness → `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]`).

E.g. for size $50$ patches the input to the linear layer is a $25$ dimensional vector:
$\frac{250}{50} \times \frac{250}{50} = 25$




