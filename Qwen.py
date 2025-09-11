from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

from TextStreamer import TextStreamer


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        print("模型加载" )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("模型加载完成")

    def generate_response(self, user_input):
        print("输入："+user_input)
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768,streamer=self.streamer)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        print("输出：" + response)
        return response
        
    def stream_generate_response(self, user_input):
        print("输入："+user_input)
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        
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
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=32768, streamer=streamer)
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
        
        # 更新历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_response})
        print("输出：" + full_response)
        return full_response  # 修复未定义变量错误，返回完整响应
