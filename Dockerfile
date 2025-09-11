FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-py311-torch2.3.1-1.29.0
LABEL authors="weiii"
RUN pip install Flask==3.1.1  -i https://mirrors.aliyun.com/pypi/simple/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
COPY . .
RUN chmod +x main.py
EXPOSE 80
CMD ["python", "main.py"]

