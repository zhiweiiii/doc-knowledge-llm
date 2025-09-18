from concurrent.futures import ThreadPoolExecutor
import queue
import os

from Qwen import QwenChatbot


class QwenThread(ThreadPoolExecutor):

    def __init__(self, **kwargs):
        super(QwenThread, self).__init__(max_workers= 1,thread_name_prefix="test_",**kwargs)
        if not os.path.exists("/model/Qwen/Qwen3-0___6B/model.safetensors"):
            print("没有发现模型文件，自动下载文件")
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download('Qwen/Qwen3-0.6B', cache_dir='/model/')
        self.qwen = QwenChatbot(model_name="/model/Qwen/Qwen3-0___6B/")
       
        
        input_message = ["下面我会问你几个问题，用来测试你的准确性，请根据我提供的文档回答", "1+1等于几？"]
        for e in input_message:
            self.qwen.generate_response(e)

    # 流式对话接口
    def stream_chat(self, text):
        # 创建一个队列来接收流式输出
        result_queue = queue.Queue()
        
        # 提交任务到线程池
        self.submit(self.stream_infer, text, result_queue)
        
        # 从队列中读取结果并yield
        while True:
            item = result_queue.get()
            if item is None:  # None表示结束
                break
            yield item
            result_queue.task_done()

    # 外部对话接口（保持兼容性）
    def chat(self,text):
        result =self.submit(self.infer, text)
        return result.result()

    def infer(self, text):
        result_str = self.qwen.generate_response(text)
        return result_str
        
    # 流式推理方法
    def stream_infer(self, text, result_queue):
        try:
            # 调用QwenChatbot的流式生成方法
            for chunk in self.qwen.stream_generate_response(text):
                if chunk:
                    result_queue.put(chunk)
        finally:
            # 发送结束信号
            result_queue.put(None)

