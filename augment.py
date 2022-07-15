import os
import cv2
import xml
import wand
import glob
import shutil
import numpy as np
from pathlib import Path
from PIL import Image as im
from wand.image import Image
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
#-------------Augmentaion Functions---------------#
def process_n_save_files(output_directory_path,input_directory_path,func_var,i):

  try:
    file_xml_path = (input_directory_path).split('.')[0] +".xml"
    file_txt_path = (input_directory_path).split('.')[0] +".txt"
    if os.path.exists(file_xml_path):
      output_directory_path_xml = output_directory_path+(input_directory_path).split('\\')[-1].split('.')[0]+func_var+str(i)+".xml"
      xml_file = shutil.copyfile(file_xml_path, output_directory_path_xml)

    if os.path.exists(file_txt_path):
      output_directory_path_txt = output_directory_path+(input_directory_path).split('\\')[-1].split('.')[0]+func_var+str(i)+".txt"
      txt_file = shutil.copyfile(file_txt_path, output_directory_path_txt) 

  except Exception as e:
    print("Exception in process_n_save_files",e)
#-------------------------------------------------#
def Add_Noise(img, input_directory_path, output_directory_path, scale):
  try:
    if type(scale)==int:
        scale = [scale]
    for i in scale: 
      scale = i*100
      img.noise("poisson", attenuate = scale)
      filename = os.path.basename("/"+(input_directory_path).split('/')[-1].split('.')[0]+"_noise_"+str(i)+".jpeg")
      img.save(filename = output_directory_path+filename)
      
      process_n_save_files(output_directory_path,input_directory_path,"_noise_",i)  
  except Exception as e:
    print("Exceptipn in Add_Noise",e)         
#-------------------------------------------------#
def Add_Blur(img, input_directory_path, output_directory_path, scale):
  try:
    if type(scale)==int:
      scale = [scale]
    for i in scale:
      img.blur(radius=0, sigma=i)
      filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_blur_"+str(i)+".jpeg")
      img.save(filename = output_directory_path+filename)
      
      process_n_save_files(output_directory_path,input_directory_path,"_blur_",i)  
  except Exception as e:
     print("Exceptipn in Add_Blur",e)
#-------------------------------------------------#
def Add_Exposure(img, input_directory_path, output_directory_path, scale):
  try:
    if type(scale)==int:
      scale = [scale] 
    for i in scale:
      scale = (i/100)*2
      img.blue_shift(factor = scale)
      filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_exposure_"+str(i)+".jpeg")
      img.save(filename = output_directory_path+filename)

      process_n_save_files(output_directory_path,input_directory_path,"_exposure_",i)
  except Exception as e:
     print("Exceptipn in Add_Exposure",e)
#-------------------------------------------------#
def Add_Hue(img, input_directory_path, output_directory_path, scale):
  try:
    img.modulate(100, 100, scale)
    filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_hue_"+str(scale)+".jpeg")
    img.save(filename = output_directory_path+filename)

    process_n_save_files(output_directory_path,input_directory_path,"_hue_",scale)
  except Exception as e:
     print("Exceptipn in Add_Hue",e)
#-------------------------------------------------#
def Add_Saturation(img, input_directory_path, output_directory_path, scale):
  try:
    img.modulate(100, scale, 100)
    filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_saturation_"+str(scale)+".jpeg")
    img.save(filename = output_directory_path+filename)

    process_n_save_files(output_directory_path,input_directory_path,"_saturation_",scale)
  except Exception as e:
     print("Exceptipn in Add_Saturation",e)
#-------------------------------------------------#
def Add_Colorize(img, input_directory_path, output_directory_path, color):
  try:
    img.colorize(color = color, alpha = "rgb(15 %, 15 %, 15 %)")
    filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_colorize_"+color+".jpeg")
    img.save(filename = output_directory_path+filename)

    process_n_save_files(output_directory_path,input_directory_path,"_colorize_",color)
  except Exception as e:
     print("Exceptipn in Add_Colorize",e)
#-------------------------------------------------#
def Add_Grayscale(img, input_directory_path, output_directory_path):
  try:
    img.type = 'grayscale';
    filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_grayscale1"+".jpeg")
    img.save(filename = output_directory_path+filename)

    process_n_save_files(output_directory_path,input_directory_path,"_grayscale",1)
  except Exception as e:
     print("Exceptipn in Add_Grayscale",e)
#-------------------------------------------------#
def Add_Sharpen(img, input_directory_path, output_directory_path, scale, scale1):
  try:
    with img.clone() as sharpen:
      sharpen.sharpen(scale , scale1)
      filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_sharpen_"+str(scale)+"_"+str(scale1)+".jpeg")
      sharpen.save(filename = output_directory_path+filename)
      i=str(scale)+"_"+str(scale1)

      process_n_save_files(output_directory_path,input_directory_path,"_sharpen_",i)
  except Exception as e:
     print("Exceptipn in Add_Sharpen",e)
#-------------------------------------------------#
def Add_Brightness(img, input_directory_path, output_directory_path, scale, scale1):
  try:
    with img.clone() as brightness_contrast:
      brightness_contrast.brightness_contrast(scale, scale1)
      filename = os.path.basename("/"+(input_directory_path).split('.')[0]+"_brighten_"+str(scale)+"_"+str(scale1)+".jpeg")
      brightness_contrast.save(filename = output_directory_path+filename)
      i=str(scale)+"_"+str(scale1)

      process_n_save_files(output_directory_path,input_directory_path,"_brighten_",i)
  except Exception as e:
     print("Exceptipn in Add_Brightness",e)
#-------------------------------------------------#
def rotate_im(image, angle):
    """Rotate the image.
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    numpy.ndarray
        Rotated Image
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image
#-------------------------------------------------#
def get_corners(bboxes): 
    """Get corners of bounding boxes
    Parameters
    ----------
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    x2 = x1 + width
    y2 = y1 
    x3 = x1
    y3 = y1 + height
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners
#-------------------------------------------------#
def rotate_box(corners,angle,  cx, cy, h, w):
    """Rotate the bounding box.    
    Parameters
    ----------
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    angle : (float) angle by which the image is to be rotated 
    cx : (int) x coordinate of the center of image (about which the box will be rotated)
    cy : (int) y coordinate of the center of image (about which the box will be rotated)
    h  : (int) height of the image
    w  : (int) width of the image
    Returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated
#-------------------------------------------------#
def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    Parameters
    ----------
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    Returns 
    -------
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final
#-------------------------------------------------#
def convert_yolo_to_voc(image, bboxes_y):
    image = cv2.imread(image)
    i_w,i_h = image.shape[:2][::-1]
    if len(bboxes_y.shape)==1:
        add_dim=[]
        add_dim.append((bboxes_y.tolist()))
        bboxes_y=np.array(add_dim)
    x_min = ((bboxes_y[:,1] - bboxes_y[:,3] / 2)*i_w)
    x_max = ((bboxes_y[:,1] + bboxes_y[:,3] / 2)*i_w)
    y_min = ((bboxes_y[:,2] - bboxes_y[:,4] / 2)*i_h)
    y_max = ((bboxes_y[:,2] + bboxes_y[:,4] / 2)*i_h)

    bboxes_v=np.array([i for i in zip(x_min,y_min,x_max,y_max)])
    return bboxes_v
#-------------------------------------------------#
def covert_voc_to_yolo(image, bboxes_v):
    image = cv2.imread(image)
    i_w,i_h = image.shape[:2][::-1]
    x_center = np.array([float((bboxes_v[:,0] + bboxes_v[:,2])) / 2 / i_w])
    y_center = np.array([float((bboxes_v[:,1] + bboxes_v[:,3])) / 2 / i_h])
    w = np.array([float((bboxes_v[:,2] - bboxes_v[:,0])) / i_w])
    h = np.array([float((bboxes_v[:,3] - bboxes_v[:,1])) / i_h])

    bboxes_y=np.array([i for i in zip(x_center,y_center,w,h)])
    return bboxes_y
#-------------------------------------------------#
def horizontal_filp_voc(image, bboxes):
    ''' Takes input as image & bounding boxes in voc format ie (xmin, ymin, xmax, ymax) & 
    return both by changing for horizontal flip in same format.
    '''
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i_w,i_h = img.shape[:2][::-1]
    img_hf = img[:, ::-1, :]

    for i in range(bboxes.shape[0]):
        box_w=int(bboxes[i][2]) - int(bboxes[i][0])
        bboxes[i][0] += int(2*(i_w/2 - int(bboxes[i][0])) - box_w)
        bboxes[i][2] += int(2*(i_w/2 - int(bboxes[i][2])) + box_w)
    return img_hf, bboxes
#-------------------------------------------------#
def vertical_filp_voc(image, bboxes):
    '''
    Takes input as image & bounding boxes in voc format ie (xmin, ymin, xmax, ymax) & 
    return both by changing for horizontal flip in same format.
    '''
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i_w,i_h = img.shape[:2][::-1]
    img_vf = img[::-1, :, :]

    for i in range(bboxes.shape[0]):
        box_h=int(bboxes[i][3]) - int(bboxes[i][1])
        bboxes[i][1] += int(2*(i_h/2 - int(bboxes[i][1])) - box_h)
        bboxes[i][3] += int(2*(i_h/2 - int(bboxes[i][3])) + box_h)
    return img_vf, bboxes
#-------------------------------------------------#
def rotation_voc(img, bboxes, angle):
    '''
    Takes input as image & bounding boxes in voc format ie (xmin, ymin, xmax, ymax) & 
    return both by changing image angle in same format.
    '''
    
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2
    
    boxes_coords = bboxes
    corners = get_corners(boxes_coords)
    boxes_coords = np.array(boxes_coords, dtype = float )

    corners = np.hstack((corners, boxes_coords[:,4:]))
    image_r = rotate_im(img, angle)
    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)

    scale_factor_x = image_r.shape[1] / w
    scale_factor_y = image_r.shape[0] / h

    image_r = cv2.resize(image_r, (w,h))

    new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

    bboxes_r  = new_bbox
    
    return image_r, bboxes_r
#-------------------------------------------------#
def draw_rect(im, cords, color = None):
    '''cords format `x1 y1 x2 y2` or `xmin ymin xmax ymax`'''
    im = im.copy()

    cords = cords[:,:4]
    cords = cords.reshape(-1,4)
    if not color:
        color = [255,255,255]
    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

def xml_to_bbox(soup):
    x_min = soup.select('xmin')
    y_min = soup.select('ymin')
    x_max = soup.select('xmax')
    y_max = soup.select('ymax')
    x_min1 = [int(i.text) for i in x_min]
    y_min1 = [int(i.text) for i in y_min]
    x_max1 = [int(i.text) for i in x_max]
    y_max1 = [int(i.text) for i in y_max]
    bboxes = np.array([i for i in zip(x_min1,y_min1,x_max1,y_max1)])
    
    return bboxes

def get_annotation_type(input_directory_path):
    try:
        if any(File.endswith(".xml") for File in os.listdir(input_directory_path)):
            annotation_type = 'VOC'
        elif any(File.endswith(".txt") for File in os.listdir(input_directory_path)):
            annotation_type = 'YOLO'
        else:
            print("Directory not contain xml or txt file of annotation")
            annotation_type = None
        return annotation_type
    except Exception as e:
        print("Exception in get_annotation_type",e)

def get_images_list(input_directory_path):
    try:
        all_images =[]
        for root, dirs, files in os.walk(input_directory_path):
            for file in files:
                if file.endswith(('.png','.jpg','.jpeg')):
                    all_images.append(os.path.join(root, file))
        return all_images
    except Exception as e:
        print("Exception in get_images_list",e)

def read_xml_file(image):        
    try:
        pre, ext = os.path.splitext(image)
        xml_file=pre+'.xml'
        with open( xml_file, "r") as file:
            contents = file.read()            
        return contents,ext
    except Exception as e:
        print("Exception in read_xml_file",e)

def process_horizontal_flip_voc(image,contents,output_directory_path,ext):
    try:
        soup = BeautifulSoup(contents, 'xml')
        bboxes = xml_to_bbox(soup)
        image_hf, bboxes_hf = horizontal_filp_voc(image, bboxes)
        cv2.imwrite(os.path.join(output_directory_path, os.path.basename(image).replace(ext,'_hf'+ext)),
                    cv2.cvtColor(image_hf, cv2.COLOR_BGR2RGB))

        for i in range(bboxes.shape[0]):
            soup.select('xmin')[i].string = str(bboxes_hf[i][0])
            soup.select('xmax')[i].string = str(bboxes_hf[i][2])

        with open(output_directory_path +'\\' + os.path.basename(image).replace(ext, "_hf.xml"), "w") as file1:
            file1.write(soup.prettify())
            file1.close()                    
    except Exception as e:
        print("Exception in process_horizontal_flip_voc",e)
    
def process_vertical_flip_voc(image,contents,output_directory_path,ext):
    try:
        soup = BeautifulSoup(contents, 'xml')
        bboxes = xml_to_bbox(soup)
        image_vf, bboxes_vf = vertical_filp_voc(image, bboxes)
        cv2.imwrite(os.path.join(output_directory_path, os.path.basename(image).replace(ext,'_vf'+ext)),
                    cv2.cvtColor(image_vf, cv2.COLOR_BGR2RGB))

        for i in range(bboxes.shape[0]):
            soup.select('ymin')[i].string = str(bboxes_vf[i][1])
            soup.select('ymax')[i].string = str(bboxes_vf[i][3])

        with open(output_directory_path +'\\' + os.path.basename(image).replace(ext, "_vf.xml"), "w") as file1:
            file1.write(soup.prettify())
            file1.close()
    except Exception as e:
        print("Exception in process_vertical_flip_voc",e)

def process_rotation_voc(image,contents,output_directory_path,ext,Rotation):
    try:
        if type(Rotation)==int:
            Rotation=[Rotation]            
        for angle in Rotation:
            soup = BeautifulSoup(contents, 'xml')
            bboxes = xml_to_bbox(soup)
            image_r, bboxes_r = rotation_voc(image, bboxes, angle)
            cv2.imwrite(os.path.join(output_directory_path, os.path.basename(image).replace(ext,'_r_'+str(angle)+ext)),
                        cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB))

            for i in range(bboxes.shape[0]):
                soup.select('xmin')[i].string = str(int(bboxes_r[i][0]))
                soup.select('xmax')[i].string = str(int(bboxes_r[i][2]))
                soup.select('ymin')[i].string = str(int(bboxes_r[i][1]))
                soup.select('ymax')[i].string = str(int(bboxes_r[i][3]))

            with open(output_directory_path +'\\' + os.path.basename(image).replace(ext, f"_r_{angle}.xml"), "w") as file1:
                file1.write(soup.prettify())
                file1.close()
        
    except Exception as e:
        print("Exception in process rotation",e)

def read_yolo_txt_file(image):    
    try:
        pre, ext = os.path.splitext(image)
        txt_file=pre+'.txt'
        bboxes_y=np.loadtxt(txt_file, dtype=float)
        if len(bboxes_y.shape)==1:
            bbox_1=[]
            bbox_1.append((bboxes_y.tolist()))
            bboxes_y=np.array(bbox_1)
        bboxes_v=convert_yolo_to_voc(image, bboxes_y)
        return bboxes_v,ext,bboxes_y
    except Exception as e:
        print("Exception in read_yolo_txt_file",e)

def process_horizontal_flip_yolo(image, bboxes_v,output_directory_path,ext,bboxes_y):
    try:
        bboxes_v_h=bboxes_v.copy()
        bboxes_y_h=bboxes_y.copy()
        image_hf, bboxes_hf_v = horizontal_filp_voc(image, bboxes_v_h)
        cv2.imwrite(os.path.join(output_directory_path, os.path.basename(image).replace(ext,'_hf'+ext)),
                    cv2.cvtColor(image_hf, cv2.COLOR_BGR2RGB))

        bboxes_hf_y=covert_voc_to_yolo(image, bboxes_hf_v)

        bbox_with_classes = np.concatenate([bboxes_y_h[:,0].reshape(-1,1), bboxes_hf_y], axis = 1)

        with open(output_directory_path +'\\' + os.path.basename(image).replace(ext, "_hf.txt"), "w") as out_file:
            for box in bbox_with_classes:
                out_file.write("%d %.8f %.8f %.8f %.8f\n" % (box[0], box[1], box[2], box[3], box[4]))

    except Exception as e:
        print("Exception in process_horizontal_flip_yolo",e)

def process_vertical_flip_yolo(image, bboxes_v,output_directory_path,ext,bboxes_y):
    try:
        bboxes_v_v=bboxes_v.copy()
        bboxes_y_v=bboxes_y.copy()

        image_vf, bboxes_vf_v = vertical_filp_voc(image, bboxes_v_v)
        cv2.imwrite(os.path.join(output_directory_path, os.path.basename(image).replace(ext,'_vf'+ext)),
                    cv2.cvtColor(image_vf, cv2.COLOR_BGR2RGB))
        bboxes_vf_y=covert_voc_to_yolo(image, bboxes_vf_v)
        bbox_with_classes = np.concatenate([bboxes_y_v[:,0].reshape(-1,1), bboxes_vf_y], axis = 1)

        with open(output_directory_path +'\\' + os.path.basename(image).replace(ext, "_vf.txt"), "w") as out_file:
            for box in bbox_with_classes:
                out_file.write("%d %.8f %.8f %.8f %.8f\n" % (box[0], box[1], box[2], box[3], box[4]))
        
    except Exception as e:
        print("Exception in process_vertical_flip_yolo",e)        

def process_rotation_yolo(image,bboxes_v,output_directory_path,ext,bboxes_y,Rotation):
    try:
        bbxes_v_r=bboxes_v.copy()
        bboxes_y_r=bboxes_y.copy()
        
        if type(Rotation)==int:
            Rotation=[Rotation]            
        for angle in Rotation:
            image_r, bboxes_r_v = rotation_voc(image, bbxes_v_r, angle)
            cv2.imwrite(os.path.join(output_directory_path, os.path.basename(image).replace(ext,'_r_'+str(angle)+ext)),
                        cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB))
            bboxes_r_y=covert_voc_to_yolo(image, bboxes_r_v)
            bbox_with_classes = np.concatenate([bboxes_y_r[:,0].reshape(-1,1), bboxes_r_y], axis = 1)

            with open(output_directory_path +'\\' + os.path.basename(image).replace(ext, f'_r_{angle}.txt'), "w") as out_file:
                for box in bbox_with_classes:
                    out_file.write("%d %.8f %.8f %.8f %.8f\n" % (box[0], box[1], box[2], box[3], box[4]))                        
        
    except Exception as e:
        print("Exception in process rotation",e)
def extract_names(input_directory_path):
  print("---within extract_names----")
  ext_list = ['*.jpeg', '*.jpg', '*.png']
  all_images = []

  for i in ext_list:
    for img in glob.glob(input_directory_path+i):
      all_images.append(img)
  return(all_images)

#-------------Extract Names---------------#
def augmentation(input_directory_path, output_directory_path, Horizontal_Flip = None, Vertical_Flip = None, Rotation = None,Noise=None, Blur=None, Exposure=None, Hue=None, Saturation=None, Colorize=None,
                Grayscale=None, Sharpen=None, Brightness=None):
    # path_Exist = os.path.exists(output_directory_path)
    # if not path_Exist: 
    #     os.makedirs(output_directory_path)
    #     print("The new directory is created!")


    annotation_type=get_annotation_type(input_directory_path)
    all_images=get_images_list(input_directory_path)
    all_images = extract_names(input_directory_path)
    for input_directory_path in all_images:
        with Image(filename = input_directory_path) as img:
            if Noise:
                Add_Noise(img, input_directory_path, output_directory_path, Noise)
                
            if Blur:
                Add_Blur(img, input_directory_path, output_directory_path, Blur)

            if Exposure:
                Add_Exposure(img, input_directory_path, output_directory_path, Exposure)

            if Hue:
                Add_Hue(img, input_directory_path, output_directory_path, Hue)

            if Saturation:
                Add_Saturation(img, input_directory_path, output_directory_path, Saturation)

            if Colorize:
                Add_Colorize(img, input_directory_path, output_directory_path, Colorize)

            if Grayscale==True:
                Add_Grayscale(img, input_directory_path, output_directory_path)

            if Sharpen:
                Add_Sharpen(img, input_directory_path, output_directory_path, Sharpen[0], Sharpen[1])

            if Brightness:
                Add_Brightness(img, input_directory_path, output_directory_path, Brightness[0], Brightness[1])
    if annotation_type:
        if annotation_type=="VOC":
            for image in all_images:                
                contents,ext=read_xml_file(image)
                if Horizontal_Flip:
                    process_horizontal_flip_voc(image,contents,output_directory_path,ext)
                    
                if Vertical_Flip:
                    process_vertical_flip_voc(image,contents,output_directory_path,ext)

                if Rotation:
                    process_rotation_voc(image,contents,output_directory_path,ext,Rotation)
        elif annotation_type=="YOLO":
            for image in all_images:
                bboxes_v,ext,bboxes_y=read_yolo_txt_file(image)

                if Horizontal_Flip:
                    process_horizontal_flip_yolo(image, bboxes_v,output_directory_path,ext,bboxes_y)
    
                if Vertical_Flip:
                    process_vertical_flip_yolo(image, bboxes_v,output_directory_path,ext,bboxes_y)
                
                if Rotation:
                    process_rotation_yolo(image,bboxes_v,output_directory_path,ext,bboxes_y,Rotation)

#augmentation('D:\\Sanket Kulkarni\\API\\tkinter_augmenations\\input\\','D:\\Sanket Kulkarni\\API\\tkinter_augmenations\\output\\', Colorize='red')


