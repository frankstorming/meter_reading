import cv2
import numpy as np
from math import cos, pi, sin, acos

#centers表示所有模板图片的指针中心点坐标，(0,0)位于图片左上角
centers = [[47,50],[67,74],[102,96],[63,64],[66,67],[65,67],[107,105],[104,106],[94,89],[57,55],[66,71]]
#scales表示所有模板图片刻度线所在坐标
scales=[{0:(8,68),1000:(7,46),2000:(14,20),3000:(38,8),4000:(64,5),5000:(84,22),6000:(98,42),7000:(96,66)},
        {0:(13,67),150:(17,50),200:(21,36),250:(30,27),300:(41,22),450:(65,18)},
        {0:(24,97),1000:(27,73),1500:(30,50),2000:(47,40),2500:(56,28),3000:(68,25),4000:(83,16),5000:(93,16),6000:(99,17)},
        {0:(14,63),50:(19,48),100:(26,29),150:(44,17),200:(65,13)},
        {0:(14,67),200:(17,46),300:(24,32),400:(36,22),500:(50,15),600:(65,13)},
        {0:(15,66),10:(14,55),20:(19,39),30:(32,36),40:(46,15),50:(63,13)},
        {0:(30,105),1:(26,86),2:(36,71),3:(40,52),4:(58,42),5:(71,27),6:(90,27),7.2:(110,19)},
        {0:(20,105),20:(21,80),40:(35,58),60:(54,39),80:(77,27),100:(101,26)},
        {0:(25,90),0.5:(24,77),1.0:(32,57),1.5:(47,38),2.0:(66,24),2.5:(91,19)},
        {0:(13,55),100:(12,45),200:(18,29),300:(29,19),400:(38,12),600:(57,10)},
        {0:(18,71),200:(16,62),400:(21,43),600:(31,26),800:(43,18),1000:(64,14)}]
#angles表示所有模板图片对应刻度相对中心点的角度
angles=[]
#模板原图大小
original_template_image_size = [(128,124),(107,105),(164,166),(99,97),(104,107),(106,99),(163,160),(161,162),(140,140),(95,104),(112,114)]

#计算各个模板图刻度对应的角度
def calculate_angles(centers, scales):
    template_number = len(centers)
    for i in range(0, template_number):
        angles.append({})
        #print(f"模板{i+1}：")
        for k, v in scales[i].items():
            #第一个模板图片为圆形表盘，以中心点为轴，→为起始边向下旋转所成角度为r,r属于(0,360)
            if i == 0:
                r = acos((v[0] - centers[i][0])/((v[0] - centers[i][0]) ** 2 + (v[1] - centers[i][1]) ** 2) ** 0.5)
                r = int(r * 180 / pi)
                if 1000 < k < 7000:
                    r = 360 - r
            else:
                r = acos((centers[i][0] - v[0])/((v[0] - centers[i][0]) ** 2 + (v[1] - centers[i][1]) ** 2) ** 0.5)
                r = int(r * 180 / pi)
            angles[i][k]=r
            #print(f"{k}刻度的角度为:",angles[i][k])

calculate_angles(centers,scales)


#根据角度和表盘类型，求得指针式仪表盘数值
def get_pointer_meter_value(angle, template_type):
    #value是所要求得指针数值，scale_value_down是刚好小于指针数值的表盘刻度数值，scale_value_over是刚好大于指针数值的表盘刻度数值
    value = 0
    scale_value_down = -1
    scale_value_up = 0
    
    #表盘为圆形
    if template_type == 0:
        if angles[template_type][0] < angle < angles[template_type][1000]:
            scale_value_down = 0
            scale_value_up = 1000
        elif angles[template_type][1000] < angle < angles[template_type][2000]:
            scale_value_down = 1000
            scale_value_up = 2000
        elif angles[template_type][2000] < angle < angles[template_type][3000]:
            scale_value_down = 2000
            scale_value_up = 3000
        elif angles[template_type][3000] < angle < angles[template_type][4000]:
            scale_value_down = 3000
            scale_value_up = 4000
        elif angles[template_type][4000] < angle < angles[template_type][5000]:
            scale_value_down = 4000
            scale_value_up = 5000
        elif angles[template_type][5000] < angle < angles[template_type][6000]:
            scale_value_down = 5000
            scale_value_up = 6000
        elif angles[template_type][7000] < angle < angles[template_type][0]:
            return 0
        else:
            angles_difference_angle = angles[template_type][7000] + 360 - angles[template_type][6000]
            if angle > angles[template_type][6000]:
                pointer_difference_angle = angle - angles[template_type][6000]
            else:
                pointer_difference_angle = angle + 360 - angles[template_type][6000]
            value = 6000 + 1000 * pointer_difference_angle / angles_difference_angle
            return value
    
    #表盘为四分之一圆
    else:    
        for k,v in angles[template_type].items():
            if angle < v:
                if k==0:
                    return 0
                else:
                    scale_value_up = k
                    if scale_value_down != -1:
                        break;
            else:
                scale_value_down = k
            
    angles_difference_angle = angles[template_type][scale_value_up] - angles[template_type][scale_value_down] #刻度线间角度差值
    pointer_difference_angle = angle - angles[template_type][scale_value_down]#下游刻度线与指针角度的差值
    value = scale_value_down + (scale_value_up-scale_value_down) * pointer_difference_angle / angles_difference_angle
    return value

#get_pointer_meter_value(270,0)
#模板匹配方法
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = cv2.TM_CCOEFF_NORMED

#图片数量
img_number = 58
#图片读数结果
meter_reading_result = [0 for i in range(img_number)]

#各个模板图片对应的图片编号,共11个模板图片，58个图片
template_index = [[1,2,44,45],
                  [3,10,23,28,52],
                  [4,5,6,7,8,9,11,12,13,24,25,26,29,30,31,32,33,34,49,50,51,53,54,55],
                  [14,35,56],
                  [15],
                  [16],
                  [17,18,19,22,36,37,38,39,42,58],
                  [20,40],
                  [21,41,43,57],
                  [27],
                  [46,47,48]]

def get_match_rect(template, img, method):
    #获取模板匹配的矩形的左上角和右下角的坐标
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(img, template, method)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的方法，对结果的解释不同
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right

#def image_resize(img,template):
 #   image_width, image_height = img.shape[1], img.shape[0]
  #  template_width, template_height = template.shape[1], template.shape[0]
   # w, h = image_width, image_height
    #result = img
    #if image_width < template_width * 1.2 or image_height < template_height * 1.2:
     #   w = template_width*1.2
      #  h = template_height*1.2
       # result = cv2.resize(img, (w,h))
    #return result

#高斯滤波除噪
def gaussian_filter(img):
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    return gaussian

#获取指定图片的指针角度
def get_pointer_angle(img, template_type):
    #shape = img.shape
    center = centers[template_type]
    center_x = center[0]
    center_y = center[1]
    freq_list = []
    #圆形表盘
    if template_type == 0:
        for i in range(361):
            x = 0.6 * center_x * cos(i * pi / 180) + center_x
            y = 0.6 * center_x * sin(i * pi / 180) + center_y
            x1 = 0.4 * center_x * cos(i * pi / 180) + center_x
            y1 = 0.4 * center_x * sin(i * pi / 180) + center_y
            temp = img.copy()
            cv2.line(temp, (int(x1), int(y1)), (int(x), int(y)), 255, thickness=2)
            freq_list.append((np.sum(temp), i))
            #cv2.imshow('get_pointer_angle', temp)
            #cv2.waitKey(10)
    else:
        for i in range(91):
            x = center_x - 0.6 * center_x * cos(i * pi / 180)
            y = center_y - 0.6 * center_x * sin(i * pi / 180)
            temp = img.copy()
            cv2.line(temp, (center_x, center_y), (int(x), int(y)), 255, thickness=2)
            freq_list.append((np.sum(temp), i))
            #cv2.imshow('get_pointer_angle', temp)
            #cv2.waitKey(30)
    #cv2.destroyAllWindows()
    freq = max(freq_list, key = lambda x:x[0])
    return freq[1]

#对于一红一黑双指针，先识别出红指针
def get_red_pointer_angle(img, template_type):
    center = centers[template_type]
    center_x = center[0]
    center_y = center[1]
    freq_list = []
    for i in range(91):
        x = center_x - 0.6 * center_x * cos(i * pi / 180)
        y = center_y - 0.6 * center_y * sin(i * pi / 180)
        temp = img.copy()
        cv2.line(temp, (center_x, center_y), (int(x), int(y)), (0, 0, 255), thickness=2)
        #cv2.imshow('red_pointer', temp)
        #cv2.waitKey(30)
        temp = np.sum(temp, axis=0)
        temp = np.sum(temp, axis=0)
        #获取图片中红色亮度的总和
        temp = temp[2]
        freq_list.append((np.sum(temp), i))
    #cv2.destroyAllWindows()
    freq = min(freq_list, key = lambda x:x[0])
    red_pointer_angle = freq[1]
    return red_pointer_angle

#展示图片刻度值
def display_result(result):
    length = len(result)
    print("---------------------------------------")
    for i in range(length):
        if type(result[i]) == tuple:
            #print(i)
            #print(result[i])
            print("{}、red:{:.2f},black:{:.2f}".format((i+1), result[i][0], result[i][1]))
        else:
            print("{}、{:.2f}".format((i+1), result[i]))
    print("---------------------------------------")

#在原图中画出指针位置，对于一红一黑指针情况，白线代表红指针，绿线代表黑指针
def draw_line(img, template_type, index, angle):
        
    center_x = centers[template_type][0]
    center_y = centers[template_type][1]
    temp = img.copy()
    if template_type == 0:
        x = 0.6 * center_x * cos(angle * pi / 180) + center_x
        y = 0.6 * center_x * sin(angle * pi / 180) + center_y
        x1 = 0.4 * center_x * cos(angle * pi / 180) + center_x
        y1 = 0.4 * center_x * sin(angle * pi / 180) + center_y
        cv2.line(temp, (int(x1), int(y1)), (int(x), int(y)), (255, 255, 255), thickness=2)
    elif template_type == 2 or template_type == 9 or template_type == 10:
        red_pointer_angle, black_pointer_angle = angle
        x1 = center_x - 0.6 * center_x * cos(red_pointer_angle * pi / 180)
        y1 = center_y - 0.6 * center_x * sin(red_pointer_angle * pi / 180)
        cv2.line(temp, (center_x, center_y), (int(x1), int(y1)), (255, 255, 255), thickness=2)
        x2 = center_x - 0.6 * center_x * cos(black_pointer_angle * pi / 180)
        y2 = center_y - 0.6 * center_x * sin(black_pointer_angle * pi / 180)
        cv2.line(temp, (center_x, center_y), (int(x2), int(y2)), (0, 255, 0), thickness=2)
    else:
        x = center_x - 0.6 * center_x * cos(angle * pi / 180)
        y = center_y - 0.6 * center_x * sin(angle * pi / 180)
        cv2.line(temp, (center_x, center_y), (int(x), int(y)), (255, 255, 255), thickness=2)
    img_path = "result/" + str(index) + ".png"
    cv2.imwrite(img_path, temp)

if __name__ == '__main__':
    for i in range(0,len(template_index)):
    #for i in range(0,1):
        template = cv2.imread("image/t%s.png" % (i+1))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('template',template)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        center = centers[i]
        for j in template_index[i]:
            img = cv2.imread("image/%s.png" % j)
            img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度图
            img_resized = cv2.resize(img_grayscale, original_template_image_size[i]) #将灰度图改变至模板原图尺寸
            top_left,bottom_right=get_match_rect(template, img_resized, method)
            cv2.rectangle(img_resized, top_left, bottom_right, 255, 2)
            #cv2.imshow('img',img_resized)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            img_new = img_resized[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]
            #cv2.imshow('new image',img_new)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            img_original_resized = cv2.resize(img, original_template_image_size[i])
            img_original_new = img_original_resized[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]
            if i == 2 or i == 9 or i == 10:
                #img_original_resized = cv2.resize(img,original_template_image_size[i])
                #img_original_new = img_original_resized[top_left[1]:bottom_right[1]+1,top_left[0]:bottom_right[0]+1]
                red_pointer_angle = get_red_pointer_angle(img_original_new, i)
                
                center_x = center[0]
                center_y = center[1]
                x1 = center_x - 0.6 * center_x * cos(red_pointer_angle * pi / 180)
                y1 = center_y - 0.6 * center_x * sin(red_pointer_angle * pi / 180)
                temp = img_new.copy()
                cv2.line(temp, (center_x, center_y), (int(x1), int(y1)), 255, thickness=2)
                #cv2.imshow("temp",temp)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                black_pointer_angle = get_pointer_angle(temp, i)
                #print(f"{j}图片红色指针角度为：",int(red_pointer_angle))
                #print("刻度为：{:.2f}".format(get_pointer_meter_value(red_pointer_angle,i)))
                #print(f"{j}图片黑色指针角度为：",int(black_pointer_angle))
                #print("刻度为：{:.2f}".format(get_pointer_meter_value(black_pointer_angle,i)))
                angle = (red_pointer_angle, black_pointer_angle)
                draw_line(img_original_new, i, j, angle)
                red_value = get_pointer_meter_value(red_pointer_angle, i)
                black_value = get_pointer_meter_value(black_pointer_angle, i)
                meter_reading_result[j-1] = (red_value, black_value)
                continue
            
            angle = get_pointer_angle(img_new, i)
            
            #将匹配结果在原图中画出来
            draw_line(img_original_new, i, j, angle)
            #cv2.imwrite(f"result/{img_index}.png",)
            #print(f"{j}图片指针角度为：",int(angle))
            #print("刻度为：{:.2f}".format(get_pointer_meter_value(angle,i)))
            meter_reading_result[j-1] = get_pointer_meter_value(angle, i)
        
    display_result(meter_reading_result)
#cv2.destroyAllWindows()