![CNN_Backbone](./asset/images/0_DETR_Panoptic.png)

##### We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (**FROM WHERE DO WE TAKE THIS ENCODED IMAGE?**)

The encoded image *d x H/32 x W/32* is the image which is the output of the ***transformer encoder***. When the final feature map from ResNet-5 block is taken the shape of the feature map is *d x H/32 x W/32* and it is converted to embeddings by flattening it on H and W, and transposing it to become, 196x256 (HW x 256) as Transformers accept such sequential embeddings. This embeddings after passed through the 6 layer encoder maintains it's shape, if 196x256, and this final encoded 196x256 is again re-arranged to form the Encoded Image which after completion of object detection is sent to a Multi-head attention layer along with bounding box embeddings.  

![CNN_Backbone](./asset/images/7_Encoded_Image.png)

###### We than along with dxN Box embeddings send the encoded Image to the Multi-Head Attention

##### We do something here to generate N x M x H/32 x W/32 maps. (**WHAT DO WE DO HERE?**)

We perform a **Multi Head Attention** with the Bounding Box embeddings and the encoded image from the Transformer encoder. 

