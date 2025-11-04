from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import logging


# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        logger.info(f"开始加载模型: {model_name}")
        
        # 检查是否有可用的GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"检测到GPU，使用设备: {device}")
        else:
            device = torch.device("cpu")
            logger.info("未检测到GPU，使用CPU")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型到指定设备
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        # 如果使用GPU但device_map未自动分配，手动移动模型到GPU
        if device.type == "cuda" and next(self.model.parameters()).device.type == "cpu":
            self.model = self.model.to(device)
        
        print(f"测试模型参数在设备: {next(self.model.parameters()).device}")
        self.device = device
        self.user_histories = {}
        logger.info("模型加载完成")
        logger.debug(f"初始化后用户历史记录字典为空，包含 {len(self.user_histories)} 个用户记录")

    def stream_generate_response_with_knowledge(self, user_input, knowledge_content=None, user_id="default"):
        logger.info(f"开始为用户 [ID: {user_id}] 流式生成响应，用户输入长度: {len(user_input)} 字符，知识库内容长度: {len(knowledge_content) if knowledge_content else 0} 字符")
        
        # 获取用户历史记录，如果不存在则创建
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
            logger.debug(f"为用户 [ID: {user_id}] 创建新的历史记录")
        
        user_history = self.user_histories[user_id]
        
        # 构建消息列表，将知识库内容单独作为system角色消息
        messages = []
        if knowledge_content:
            messages.append({"role": "system", "content": f"基于以下知识库内容回答问题：\n{knowledge_content}"})
        
        # 添加用户问题
        messages.append({"role": "user", "content": user_input})
        
        # 添加历史记录（不包含之前的system消息）
        # 只添加用户和助手的对话历史
        for msg in user_history:
            if msg["role"] != "system":
                messages.insert(-1, msg)
        logger.info(f"上下文内容: {str(messages)}")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        logger.debug(f"应用聊天模板后文本长度: {len(text)} 字符")

        inputs = self.tokenizer(text, return_tensors="pt")
        logger.debug(f"Tokenize后输入形状: {inputs.input_ids.shape}")
        
        # 创建自定义的流式输出收集器，确保移除多余格式和前缀
        class StreamingGenerator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.token_cache = []
                self.print_len = 0
                self.final_text = ""
                self.new_text_callback = None
            
            def put(self, value):
                # 处理token
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]
                
                # 添加新token到缓存
                self.token_cache.extend(value.tolist())
                text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
                
                # 处理新生成的文本
                if len(text) > self.print_len:
                    new_text = text[self.print_len:]
                    self.print_len = len(text)
                    self.final_text = text
                    # 通过回调函数返回新文本，确保移除任何多余的格式或前缀
                    # 确保正确处理所有字符编码
                    if self.new_text_callback:
                        # 确保new_text是字符串类型
                        if isinstance(new_text, bytes):
                            try:
                                new_text = new_text.decode('utf-8', errors='replace')
                            except:
                                pass
                        self.new_text_callback(new_text)
            
            def end(self):
                pass
        
        # 创建一个队列来收集流式输出
        from queue import Queue
        output_queue = Queue()
        
        # 创建流式生成器
        streamer = StreamingGenerator(self.tokenizer)
        streamer.new_text_callback = lambda text: output_queue.put(text)
        
        # 在后台生成响应
        import threading
        def generate_task():
            # 将输入数据移动到模型所在的设备
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                self.model.generate(**inputs_on_device, max_new_tokens=32768, streamer=streamer)
            # 生成完成后放入None作为结束信号
            output_queue.put(None)
        
        # 启动生成线程
        thread = threading.Thread(target=generate_task)
        thread.start()
        
        # 收集完整的响应文本
        full_response = ""
        
        # 从队列中获取流式输出并yield，确保不包含多余的格式和前缀
        while True:
            chunk = output_queue.get()
            if chunk is None:  # 结束信号
                break
            full_response += chunk
            yield chunk
            output_queue.task_done()
        
        # 等待生成线程完成
        thread.join()
        
        # 更新用户历史记录，只保存用户和助手的对话
        user_history.append({"role": "user", "content": user_input})
        # user_history.append({"role": "assistant", "content": full_response})
        
        logger.info(f"流式响应生成完成，完整响应长度: {len(full_response)} 字符")
        
        return full_response
        
    def _format_history(self, history, max_content_length=10000):
        """格式化历史记录，避免日志过于冗长"""
        if not history:
            return "[]"
        
        formatted = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # 截断过长的内容
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            formatted.append(f"{{'role': '{role}', 'content': '{content}'}}")
        
        return "[" + ", ".join(formatted) + "]"
