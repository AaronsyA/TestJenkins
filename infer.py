from utils import predict

import paddlenlp as ppnlp
import paddle
from paddlenlp.datasets import load_dataset

import paddle.nn.functional as F
from utils import evaluate
from paddlenlp.transformers import LinearDecayWithWarmup
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader


data = [
    {"text":'这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'},
    {"text":'怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'},
    {"text":'作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'},
{"text":'这产品也太垃圾了吧，差评'},
{"text":'可以不要再欺骗消费者了吗！？'},
{"text":'这款产品真的好用！！推荐！'},
    {"text":'你不对劲'},
    {"text":'真的是很好用呢'}

]
label_map = {0: 'negative', 1: 'positive'}



train_ds, dev_ds, test_ds = load_dataset(
    "chnsenticorp", splits=["train", "dev", "test"])
print("sdasddasas")
model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained("checkpoint", num_classes=len(train_ds.label_list))
print("adsad")
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained("checkpoint")

print("load successful")


results = predict(
    model, data, tokenizer, label_map, batch_size=1)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text, results[idx]))