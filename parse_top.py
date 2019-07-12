__author__ = 'jmh081701'
#将top文件中的核心数据提取出来
import  json
import  random
import  copy
json_template={
    'size':{'width':"960",'depth':"3","height":"540"},
    'filename':'0.jpg',
    "object":[]
}
object_template={
                "bndbox":
                {
                    "xmin":"",#左上角
                    "ymin":"",#左上角
                    "xmax":"",#右下角
                    "ymax":""
                },
                "name":"pedestrain",
                "id"  :"1"
        }
#name : 表示需要检测的目标的名字
#id   : 给这个目标编号一个id
#name 和 id 后面需要和 label_map.pbtxt保持一致

def read_top(file):
    with open(file,'r') as fp:
        infos=[]
        info = copy.deepcopy(json_template)
        lines = fp.readlines()
        lastframe=0
        for line in lines:
            line = line.strip().split(",")
            frameNumber = int(line[1])
            xmin=abs(float(line[-4])-2)/2
            ymin=abs(float(line[-3])-2)/2
            xmax=abs(float(line[-2])-2)/2
            ymax=abs(float(line[-1])-2)/2

            if lastframe!=frameNumber:
                #新的帧
                if len(info['object']) > 0:
                    infos.append(copy.deepcopy(info))
                    print(info['filename'])
                    #保存每一帧的文件名
                info = copy.deepcopy(json_template)
                lastframe = frameNumber
                info['filename']="%s.jpg"% frameNumber
            object_ = copy.deepcopy(object_template)
            object_['bndbox']['xmin']=copy.deepcopy(xmin)
            object_['bndbox']['xmax']=copy.deepcopy(xmax)
            object_['bndbox']['ymin']=copy.deepcopy(ymin)
            object_['bndbox']['ymax']=copy.deepcopy(ymax)
            info['object'].append(copy.deepcopy(object_))


        #parse over.
        #with open('json','w') as fp2:
        #    #保存为json为文件
        #    json.dump(infos,fp2)
        return  infos
if __name__ == '__main__':
    infos=read_top(r"TownCentre-groundtruth.top")
    train_fp=open('pedestrain_train.csv','w')
    valid_fp=open('pedestrain_valid.csv','w')
    train_fp.writelines("class,filename,height,width,xmax,xmin,ymax,ymin\n")
    valid_fp.writelines("class,filename,height,width,xmax,xmin,ymax,ymin\n")
    for info in infos:
        filename = info['filename']
        height = int(info['size']['height'])
        width = int(info['size']['width'])
        print(filename)
        if random.random()<0.05:
            #划分数据集
            fp = valid_fp
        else:
            fp = train_fp
        for object_ in info['object']:
            xmin=object_['bndbox']['xmin']
            xmax=object_['bndbox']['xmax']
            ymin=object_['bndbox']['ymin']
            ymax=object_['bndbox']['ymax']
            name=object_['name']
            fp.writelines("%s,%s,%d,%d,%s,%s,%s,%s\n"%(name,filename,height,width,xmax,xmin,ymax,ymin))

    train_fp.close()
    valid_fp.close()



