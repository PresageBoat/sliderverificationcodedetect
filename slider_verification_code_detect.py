import cv2 as cv
import numpy as np
from typing import List

class SliderVerificationCodeDetect(object):
    """Slider verification code target position detection.

    If the image is opencv image, it is expected
    to have [H,W,C] shape

    Find the target position of the slider.

    Args:
        img_t (opencv image -> np.array): Slider template image.

        img_f (opencv image -> np.array): Background image without slider image.

    """

    def __init__(self) -> None:
        super().__init__()
        pass

    @staticmethod
    def templateimg_process(img:np.array)->np.array:
        kernel = np.ones((8, 8), np.uint8)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        width, heigth = gray.shape
        for h in range(heigth):
            for w in range(width):
                if gray[w, h] == 0:
                    gray[w, h] = 255
        binary = cv.inRange(gray, 255, 255)
        #先膨胀，再腐蚀-->开运算
        res = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # 
        return res

    def __call__(self,img_t:np.array,img_bg:np.array,upper_left:List[int]) ->List[int]:
        """Get location form slide verification code images.

        Args:
            img_t (opencv image): Input template image.
            img_bg (opencv image): Input background image.
            upper_left (xmin,ymin)(list of int): The upper-left point of the 
            template image which in the background image
        Returns:
            list: params (xmin, ymin, xmax, ymax)  The upper-left and 
            lower-right coordinate points of the target
             position of the slider.
        """
        if not isinstance(upper_left, list):
            raise TypeError("Argument upper_left should be int")

        if img_t is None:
            raise ValueError('Input template image is null')

        if img_bg is None:
            raise ValueError('Input background image is null')

        img_cpy=img_bg[::]
        T_height,_,_=img_t.shape
        img_crop=img_cpy[upper_left[1]-5:upper_left[1]+T_height+5,:,:]
        #pre process backgound image
        blurred = cv.GaussianBlur(img_crop, (3, 3), 0)  #
        blurred_gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        # _, target = cv.threshold(blurred_gray, 127, 255, cv.THRESH_BINARY)
        _, target = cv.threshold(blurred_gray, 0, 255, cv.THRESH_OTSU)

        cv.imwrite('blurred_gray.png', target)

        #pre process template image
        img_t_d=self.templateimg_process(img_t)

        cv.imwrite('img_t_d.png', img_t_d)


        #match template image
        method = cv.TM_CCOEFF_NORMED
        width, height = img_t_d.shape[:2]
        result = cv.matchTemplate(target, img_t_d, method)
        _, _, _, max_loc = cv.minMaxLoc(result)

        left_up =list(max_loc) 
        left_up[1]=left_up[1]+upper_left[1]
        left_up=tuple(left_up)
        right_down = (left_up[0] + height, left_up[1] + width)

        return left_up[0],left_up[1],right_down[0],right_down[1]





def test():
    def cv_imread(filepath):
        #按照BGR三通道读取图像
        cv_img=cv.imdecode(np.fromfile(filepath,dtype=np.uint8),1)
        # cv_img=cv.cvtColor(cv_img,cv.COLOR_RGB2BGR)
        return cv_img

    img_dir="./test/"
    img_path=img_dir+"3/下载.png" #背景图
    img_tp_path=img_dir+"3/p2.png"#模板图
    img=cv_imread(img_path)
    img_tp=cv_imread(img_tp_path)

    #usage
    upper_left=[0,28]
    SVCD=SliderVerificationCodeDetect()
    location=SVCD(img_tp,img,upper_left)
    cv.rectangle(img, (location[0],location[1]), (location[2],location[3]), (0, 0, 255), 2)
    cv.imwrite('match_res.png', img)

if __name__=="__main__":
    test()
    