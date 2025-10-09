import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import os
import torch
import time
import sys

from Qwen import QwenChatbot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenThread(ThreadPoolExecutor):

    def __init__(self, **kwargs):
        super(QwenThread, self).__init__(max_workers= 1,thread_name_prefix="test_",**kwargs)
        if not os.path.exists("/model/Qwen/Qwen3-0___6B/model.safetensors"):
            print("没有发现模型文件，自动下载文件")
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download('Qwen/Qwen3-0.6B', cache_dir='/model/')
        self.qwen = QwenChatbot(model_name="/model/Qwen/Qwen3-0___6B/")
       
        
        input_message = ["测试", "1+1等于几？"]
        for e in input_message:
            self.qwen.generate_response(e)

    # 流式对话接口
    def stream_chat_with_knowledge(self, text, knowledge_content=None, user_id="default"):
        logger.info(f"QwenThread接收到用户 [ID: {user_id}] 的流式聊天请求，文本长度: {len(text)} 字符，知识库内容长度: {len(knowledge_content) if knowledge_content else 0} 字符")
        
        # 创建队列存储流式结果
        result_queue = Queue()
        
        def task():
            try:
                logger.debug(f"启动流式任务，用户输入: {text[:100]}...，知识库内容: {'有内容' if knowledge_content else '无'}")
                # 使用新的支持分开传递知识库的方法
                response_iterator = self.qwen.stream_generate_response_with_knowledge(text, knowledge_content, user_id)
                for chunk in response_iterator:
                    result_queue.put(chunk)
                logger.info(f"流式响应处理完成，已将所有块放入队列")
            except Exception as e:
                logger.error(f"流式响应生成过程中出错: {str(e)}")
                error_message = f"发生错误: {str(e)}"
                result_queue.put(error_message)
            finally:
                # 标记结束
                result_queue.put(None)
                logger.debug(f"流式任务已完成，队列已关闭")
        
        # 提交任务到线程池
        future = self.submit(task)
        
        # 从队列中读取结果并yield
        response_text = ""
        while True:
            chunk = result_queue.get()
            if chunk is None:
                break
            response_text += chunk
            yield chunk
        
        logger.info(f"QwenThread流式聊天请求处理完成，生成响应长度: {len(response_text)} 字符")
        return response_text
        
    def chat_with_knowledge(self, text, knowledge_content=None, user_id="default"):
        logger.info(f"QwenThread接收到用户 [ID: {user_id}] 的同步聊天请求，文本长度: {len(text)} 字符，知识库内容长度: {len(knowledge_content) if knowledge_content else 0} 字符")
        
        def task():
            try:
                logger.debug(f"启动同步任务，用户输入: {text[:100]}...，知识库内容: {'有内容' if knowledge_content else '无'}")
                # 使用新的支持分开传递知识库的方法
                return self.qwen.generate_response_with_knowledge(text, knowledge_content, user_id)
            except Exception as e:
                logger.error(f"同步响应生成过程中出错: {str(e)}")
                return f"发生错误: {str(e)}"
        
        # 提交任务到线程池并等待结果
        future = self.submit(task)
        response = future.result()
        
        logger.info(f"QwenThread同步聊天请求处理完成，生成响应长度: {len(response)} 字符")
        return response
    
    def stream_chat(self, text, user_id):
        # 创建一个队列来接收流式输出
        result_queue = Queue()
        
        # 提交任务到线程池，传入user_id
        self.submit(self.stream_infer, text, result_queue, user_id)
        
        # 从队列中读取结果并yield
        while True:
            item = result_queue.get()
            if item is None:  # None表示结束
                break
            yield item
            result_queue.task_done()

    # 外部对话接口（保持兼容性）
    def chat(self, text, user_id="default"):
        result = self.submit(self.infer, text, user_id)
        return result.result()

    def infer(self, text, user_id="default"):
        result_str = self.qwen.generate_response(text, user_id)
        return result_str
        
    # 流式推理方法
    def stream_infer(self, text, result_queue, user_id="default"):
        try:
            # 调用QwenChatbot的流式生成方法，并传入user_id
            for chunk in self.qwen.stream_generate_response(text, user_id):
                if chunk:
                    result_queue.put(chunk)
        finally:
            # 发送结束信号
            result_queue.put(None)

