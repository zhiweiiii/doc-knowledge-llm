import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import os
import torch

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
        # if not os.path.exists("/model/Qwen/Qwen3-0___6B/model.safetensors"):
        #     print("没有发现模型文件，自动下载文件")
        #     from modelscope.hub.snapshot_download import snapshot_download
        #     snapshot_download('Qwen/Qwen3-0.6B', cache_dir='/model/')
        # self.qwen = QwenChatbot(model_name="/model/Qwen/Qwen3-0___6B/")


        if not os.path.exists("/model/Qwen/Qwen3-8B/model.safetensors.index.json"):
            print("没有发现模型文件，自动下载文件")
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download('Qwen/Qwen3-8B', cache_dir='/model/')
        self.qwen = QwenChatbot(model_name="/model/Qwen/Qwen3-8B/")
        
        

    # 流式对话接口
    def stream_chat_with_knowledge(self, text, knowledge_content=None, user_id="default"):
        logger.info(f"QwenThread接收到用户 [ID: {user_id}] 的流式聊天请求，文本长度: {len(text)} 字符，知识库内容长度: {len(knowledge_content) if knowledge_content else 0} 字符")
        
        # 创建队列存储流式结果
        result_queue = Queue()
        
        def task():
            try:
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

