from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
sam = sam_model_registry["vit_h"](checkpoint="/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D/ego4d_data/sam_vit_h_4b8939.pth")
sam.eval()
sam.cuda()
mask_generator = SamAutomaticMaskGenerator(sam)
img = cv2.imread("/home/goku/Pictures/Screenshot from 2023-04-25 16-40-16.png")
print(img.shape)
print(type(img))
print(img.dtype)
masks = mask_generator.generate(img)

# print(masks)