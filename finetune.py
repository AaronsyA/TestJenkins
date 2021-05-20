
import paddlenlp as ppnlp
import paddle
from paddlenlp.datasets import load_dataset

import paddle.nn.functional as F
from utils import evaluate
from paddlenlp.transformers import LinearDecayWithWarmup
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader

train_ds, dev_ds, test_ds = load_dataset(
    "chnsenticorp", splits=["train", "dev", "test"])

print(train_ds.label_list)
print(len(train_ds.data))
for data in train_ds.data[:5]:
    print(data)


MODEL_NAME = "ernie-1.0"

#ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=len(train_ds.label_list))

tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

# 模型运行批处理大小
batch_size = 32
max_seq_length = 128

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)


# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01


num_training_steps = len(train_data_loader) * epochs
print("ite: "+str(num_training_steps))
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()


global_step = 0
for epoch in range(1, epochs + 1):
    print("epoch: "+str(epoch))
    for step, batch in enumerate(train_data_loader, start=1):
        print("start current step  "+str(step))
        input_ids, segment_ids, labels = batch
        #labels64 = paddle.cast(labels, dtype='int64')
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
        global_step += 1
        if global_step % 10 == 0 :
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
            print("start saving..."
                  "")
            model.save_pretrained('checkpoint')
            tokenizer.save_pretrained('checkpoint')
            print("done saving...")
            #evaluate(model, criterion, metric, dev_data_loader)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
        print("current step done"+str(step))


    #



