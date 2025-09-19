

import json
import numpy as np
from modelscope.hub.snapshot_download import snapshot_download
from datetime import datetime
from safetensors.torch import save_file
import torch
import os
from transformers import AutoTokenizer, AutoModel

class VectorDatabase:
    def __init__(self, session_id):
        """初始化向量数据库"""
        self.session_id = session_id
        print("flag1")
        self.db_path = os.path.join('vector_db', session_id)
        print("flag2")
        os.makedirs(self.db_path, exist_ok=True)
        

        # 初始化嵌入模型 - 使用modelscope的文本嵌入模型
        # 首先尝试从本地加载模型，如果不存在则下载
        local_model_path = '/model/damo/nlp_corom_sentence-embedding_english-base'
        
        # if not os.path.exists(local_model_path):
        #     print(f"本地模型不存在，正在下载到 {local_model_path}")
        #     from modelscope.hub.snapshot_download import snapshot_download
        #     local_model_path = snapshot_download('damo/nlp_corom_sentence-embedding_english-base', cache_dir='/model')
        #     state_dict = torch.load(os.path.join(local_model_path, 'pytorch_model.bin'), map_location='cpu')
        #     save_file(state_dict, os.path.join(local_model_path, 'model.safetensors'))
        # else:
        #     print(f"使用本地模型: {local_model_path}")
        
        # 使用AutoTokenizer.from_pretrained和AutoModel.from_pretrained加载模型
        
        # 加载tokenizer和模型
        print(f"开始读取: {local_model_path}")
        # self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        # self.model = AutoModel.from_pretrained(local_model_path)
        # self.model.to('cpu')
        # self.model.eval()
        print(f"读取完成: {local_model_path}")
        # 创建自定义的嵌入pipeline函数
        def embedding_pipeline(text):
            # 使用tokenizer编码文本
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # 将输入移至CPU
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 提取嵌入向量（通常使用最后一层隐藏状态的平均值）
            # 不同模型可能有不同的提取方式，这里尝试几种常见方式
            if hasattr(outputs, 'sentence_embedding'):
                embedding = outputs.sentence_embedding
            elif hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # 取最后一层隐藏状态的平均值作为句子嵌入
                embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                # 如果以上都没有，使用模型输出的第一个元素（兜底方案）
                embedding = list(outputs.values())[0].mean(dim=1)
            
            # 转换为numpy数组
            embedding = embedding.cpu().numpy().squeeze()
            
            # 返回与原pipeline相同的格式
            return {'sentence_embedding': embedding}
        
        # 设置自定义的embedding_pipeline
        self.embedding_pipeline = embedding_pipeline
        print(f"成功使用AutoTokenizer和AutoModel加载模型并创建嵌入pipeline")
       
        # 初始化向量存储
        self.dimension = 768  # modelscope模型的输出维度
        self.vectors = np.array([])  # 存储所有向量
        
        # 存储文档块和元数据
        self.document_chunks = []
        self.document_metadata = []
        
        # 尝试加载已有的向量数据库
        self.load()
    
    def load(self):
        """从文件加载向量数据库"""
        vectors_path = os.path.join(self.db_path, 'vectors.npy')
        chunks_path = os.path.join(self.db_path, 'document_chunks.json')
        metadata_path = os.path.join(self.db_path, 'document_metadata.json')
        
        if os.path.exists(vectors_path) and os.path.exists(chunks_path) and os.path.exists(metadata_path):
            try:
                self.vectors = np.load(vectors_path)
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.document_chunks = json.load(f)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
                print(f"成功加载向量数据库，包含 {len(self.document_chunks)} 个文档块")
            except Exception as e:
                print(f"加载向量数据库失败: {str(e)}")
    
    def save(self):
        """保存向量数据库到文件"""
        vectors_path = os.path.join(self.db_path, 'vectors.npy')
        chunks_path = os.path.join(self.db_path, 'document_chunks.json')
        metadata_path = os.path.join(self.db_path, 'document_metadata.json')
        
        try:
            if len(self.vectors) > 0:
                np.save(vectors_path, self.vectors)
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_chunks, f, ensure_ascii=False, indent=2)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
            print(f"成功保存向量数据库，包含 {len(self.document_chunks)} 个文档块")
        except Exception as e:
            print(f"保存向量数据库失败: {str(e)}")
    
    def split_text(self, text, chunk_size=500, chunk_overlap=50):
        """将文本分块"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            # 尝试在句子边界处分割
            if end < text_length:
                # 寻找最近的句号、问号、感叹号或换行符
                punctuation_positions = [text.rfind(p, start, end) for p in ['.', '?', '!', '\n']]
                punctuation_positions = [pos for pos in punctuation_positions if pos > start + chunk_size * 0.5]  # 确保至少分割了一半
                
                if punctuation_positions:
                    end = max(punctuation_positions) + 1  # +1 包含标点符号
            
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap  # 设置下一个块的起始位置，包含重叠
        
        return chunks
    
    def add_document(self, filename, content):
        """添加文档到向量数据库"""
        # 检查文件是否已经存在
        existing_indices = [i for i, meta in enumerate(self.document_metadata) if meta['filename'] == filename]
        if existing_indices:
            # 删除已存在的文档块
            for i in sorted(existing_indices, reverse=True):
                del self.document_chunks[i]
                del self.document_metadata[i]
            
            # 重新构建向量存储
            if self.document_chunks:
                embeddings = [self.embedding_pipeline(chunk)['sentence_embedding'] for chunk in self.document_chunks]
                self.vectors = np.array(embeddings)
            else:
                self.vectors = np.array([])
        
        # 分块处理文档内容
        chunks = self.split_text(content)
        if not chunks:
            return False
        
        # 为每个文档块生成嵌入向量
        embeddings = [self.embedding_pipeline(chunk)['sentence_embedding'] for chunk in chunks]
        
        # 添加到向量存储
        if len(self.vectors) == 0:
            self.vectors = np.array(embeddings)
        else:
            self.vectors = np.vstack([self.vectors, np.array(embeddings)])
        
        # 保存文档块和元数据
        current_time = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            self.document_chunks.append(chunk)
            self.document_metadata.append({
                'filename': filename,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'timestamp': current_time
            })
        
        # 保存数据库
        self.save()
        
        return True
    
    def search(self, query, top_k=5):
        """搜索与查询最相关的文档块"""
        if len(self.document_chunks) == 0:
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_pipeline(query)['sentence_embedding']
        
        # 计算查询向量与所有存储向量的L2距离
        distances = np.linalg.norm(self.vectors - query_embedding, axis=1)
        
        # 获取距离最小的top_k个索引
        k = min(top_k, len(self.document_chunks))
        indices = np.argsort(distances)[:k]
        
        # 构建结果列表
        results = []
        for idx in indices:
            results.append({
                'content': self.document_chunks[idx],
                'metadata': self.document_metadata[idx],
                'distance': float(distances[idx])
            })
        
        return results
    
    def get_all_documents(self):
        """获取所有文档的元数据"""
        documents = {}
        for i, meta in enumerate(self.document_metadata):
            filename = meta['filename']
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'chunk_count': sum(1 for m in self.document_metadata if m['filename'] == filename),
                    'upload_time': meta['timestamp']
                }
        
        return list(documents.values())
    
    def delete_document(self, filename):
        """删除指定文档"""
        # 找出所有属于该文档的块
        indices_to_remove = [i for i, meta in enumerate(self.document_metadata) if meta['filename'] == filename]
        
        if not indices_to_remove:
            return False
        
        # 删除文档块和元数据
        for i in sorted(indices_to_remove, reverse=True):
            del self.document_chunks[i]
            del self.document_metadata[i]
        
        # 重新构建向量存储
        if self.document_chunks:
            embeddings = [self.embedding_pipeline(chunk)['sentence_embedding'] for chunk in self.document_chunks]
            self.vectors = np.array(embeddings)
        else:
            self.vectors = np.array([])
        
        # 保存数据库
        self.save()
        
        return True