import cv2
import matplotlib.pyplot as plt

# def mask_generate(I,k,grey_area):
#     shape1=I.shape[0]//(2**k)
#     shape2=I.shape[1]//(2**k)
#     for i in range(len(grey_area)):
#         row=grey_area[i]//(2**k)
#         col=grey_area[i]-row*(2**k)
#         if col==0:
#             col=2**k
#         col-=1
#         startr=row*shape1;endr=(row+1)*shape1
#         startc=col*shape2;endc=(col+1)*shape2
#         I.copy()[startr:endr,startc:endc,:]=0
#     return I



import cv2
import matplotlib.pyplot as plt
def mask_generate(I,k,grey_area):
    I_copy = I.copy()
    shape1=I.shape[0]//(2**k)
    shape2=I.shape[1]//(2**k)
    for i in range(len(grey_area)):
        row=grey_area[i]//(2**k)
        col=grey_area[i]-row*(2**k)
        if col==0:
            col=2**k
        col-=1
        startr=row*shape1;endr=(row+1)*shape1
        startc=col*shape2;endc=(col+1)*shape2
        
        I_copy[startr:endr,startc:endc,:]=0
    return I_copy


#%%
k=1     ### 2^(2*k)
grey_area = [1]  ###block number
I=cv2.imread('1f8f08ea-b5b3-4f68-94d4-3cc071b7dce8.png')   ###path
plt.imshow(I)
plt.show()
I_m=mask_generate(I,k,grey_area)
plt.imshow(I_m)
plt.show()
