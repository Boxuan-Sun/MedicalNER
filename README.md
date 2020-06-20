# MedicalNER
中文电子病例命名实体识别项目,主要实现使用了基于字向量的四层双向LSTM与CRF模型的网络

# 项目介绍

电子病历结构化是让计算机理解病历、应用病历的基础。基于对病历的结构化，可以计算出症状、疾病、药品、检查检验等多个知识点之间的关系及其概率，构建医疗领域的知识图谱，进一步优化医生的工作. CCKS2018的电子病历命名实体识别的评测任务，是对于给定的一组电子病历纯文本文档，识别并抽取出其中与医学临床相关的实体，并将它们归类到预先定义好的类别中。组委会针对这个评测任务，提供了600份标注好的电子病历文本，共需识别含解剖部位、独立症状、症状描述、手术和药物五类实体。 领域命名实体识别问题自然语言处理中经典的序列标注问题, 本项目是运用深度学习方法进行命名实体识别的一个尝试.

# 数据处理
一, 目标序列标记集合 O非实体部分,TREATMENT治疗方式, BODY身体部位, SIGN疾病症状, CHECK医学检查, DISEASE疾病实体
二, 序列标记方法 采用BIO三元标记
```
self.class_dict ={
                 'O':0,
                 'TREATMENT-I': 1,
                 'TREATMENT-B': 2,
                 'BODY-B': 3,
                 'BODY-I': 4,
                 'SIGNS-I': 5,
                 'SIGNS-B': 6,
                 'CHECK-B': 7,
                 'CHECK-I': 8,
                 'DISEASE-I': 9,
                 'DISEASE-B': 10
                }```
三, 数据转换 评测方提供了四个目录(一般项目, 出院项目, 病史特点, 诊疗经过),四个目录下有txtoriginal文件和txt标注文件,内容样式如下:

一般项目-1.txtoriginal.txt
```
女性，88岁，农民，双滦区应营子村人，主因右髋部摔伤后疼痛肿胀，活动受限5小时于2016-10-29；11：12入院。```
一般项目-1.txt:
```
右髋部	21	23	身体部位
疼痛	27	28	症状和体征
肿胀	29	30	症状和体征```
转换函数：
```
 def transfer(self):
    f = open(self.train_filepath, 'w+')
    count = 0
    for root,dirs,files in os.walk(self.origin_path):
        for file in files:
            filepath = os.path.join(root, file)
            if 'original' not in filepath:
                continue
            label_filepath = filepath.replace('.txtoriginal','')
            print(filepath, '\t\t', label_filepath)
            content = open(filepath).read().strip()
            res_dict = {}
            for line in open(label_filepath):
                res = line.strip().split('	')
                start = int(res[1])
                end = int(res[2])
                label = res[3]
                label_id = self.label_dict.get(label)
                for i in range(start, end+1):
                    if i == start:
                        label_cate = label_id + '-B'
                    else:
                        label_cate = label_id + '-I'
                    res_dict[i] = label_cate

            for indx, char in enumerate(content):
                char_label = res_dict.get(indx, 'O')
                print(char, char_label)
                f.write(char + '\t' + char_label + '\n')
    f.close()
    return```
模型输出样式：
```
，	O
男	O
，	O
双	O
塔	O
山	O
人	O
，	O
主	O
因	O
咳	SIGNS-B
嗽	SIGNS-I
、	O
少	SIGNS-B
痰	SIGNS-I
1	O
个	O
月	O
，	O
加	O
重	O
3	O
天	O
，	O
抽	SIGNS-B
搐	SIGNS-I```
