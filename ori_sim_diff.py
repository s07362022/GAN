import cv2 
import numpy as np
test_img = cv2.imread("E:\\workspace\\project_\\dimg\\186_.jpg",0) 
similar_img= cv2.imread("E:\\workspace\\project_\\dimg\\186.jpg",0) 
test_img  = test_img [:,:,None]
similar_img =similar_img[:,:,None]
test_img = cv2.resize(test_img,(300,300)) /255.0
similar_img = cv2.resize(similar_img,(300,300))/255.0

np_residual = test_img.reshape(300,300,1) - similar_img.reshape(300,300,1)
np_residual = (np_residual + 2)/2

np_residual = (255*np_residual).astype(np.uint8)
# original_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
#　similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
original_x = (test_img.reshape(300,300,1)*255.0).astype(np.uint8)
similar_x = (similar_img.reshape(300,300,1)*255.0).astype(np.uint8)

original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_COOL) #cv2.COLORMAP_JET COLORMAP_COOL
show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)
qurey = original_x
print(qurey.shape)
pred = similar_x
diff = show

import matplotlib.pyplot as plt
plt.figure(1, figsize=(3, 3))
plt.title('query image')
plt.imshow(qurey.reshape(300,300), cmap=plt.cm.gray)

plt.figure(2, figsize=(3, 3))
plt.title('generated similar image')
plt.imshow(pred.reshape(300,300), cmap=plt.cm.gray)

plt.figure(3, figsize=(3, 3))
plt.title('anomaly detection')
plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
plt.show()
