FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

## 把当前文件夹里的文件构建到镜像的根目录下, 注意 . 和 / 间是有个空格的
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /code

RUN pip install --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r ../requirements.txt

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]

