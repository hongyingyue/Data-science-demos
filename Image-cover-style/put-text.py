
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def paint_chinese_opencv(im,chinese,pos,color,font):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

    fillColor = color #(255,0,0)
    position = pos #(100,100)

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img


def process(img,text1=u'汽车与',text2=u'人工智能'):
    # step1: resize to (800,400)
    img=cv2.resize(img,(800,400))

    # text1:
    bgr_color = (0, 0, 0)
    text_color=(255,255,255)

    font = ImageFont.truetype("./仿宋.ttf", 55)
    text_width, text_height = font.getsize(text1)

    text_offset_x = 70  # start x
    text_offset_y = 70  # start y

    box_coords = ((text_offset_x, text_offset_y-10), (text_offset_x + text_width + 2, text_offset_y + text_height + 10))
    cv2.rectangle(img, box_coords[0], box_coords[1], color=bgr_color, thickness=cv2.FILLED)

    img=paint_chinese_opencv(img,text1,(text_offset_x, text_offset_y),text_color,font)

    # text2:
    bgr2_color=(0,97,255) #(255, 153, 0)
    text2_color=(0,0,0)

    text2_width,text2_height=font.getsize(text2)

    text2_offset_x=text_offset_x+text_width+4
    text2_offset_y=70

    box2_coords=((text2_offset_x,text2_offset_y-10),(text2_offset_x+text2_width+2,text2_offset_y+text2_height+10))
    cv2.rectangle(img,box2_coords[0],box2_coords[1],color=bgr2_color,thickness=cv2.FILLED)

    img=paint_chinese_opencv(img,text2,(text2_offset_x,text2_offset_y),text2_color,font)

    # text3
    cv2.putText(img, 'YueTan', (608,345), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255,255,255), thickness=2)

    # save and show

    cv2.imwrite('out.jpg',img)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__=='__main__':
    img_dir='./1.jpg'

    img=cv2.imread(img_dir)
    process(img)

