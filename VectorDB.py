

import json
import numpy as np
from modelscope.hub.snapshot_download import snapshot_download
from datetime import datetime
from safetensors.torch import save_file
import torch
import os
import logging
from transformers import AutoTokenizer, AutoModel

# 配置日志 - 设置为DEBUG级别以显示详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        """初始化向量数据库"""
        # 初始化时不绑定session_id，而是在add_document等方法中传入
        # 保存不同session的数据库路径
        self.session_paths = {}
        
        # 存储不同session的向量数据库
        self.vector_stores = {}
        
        # 初始化模型
        self._init_model()
        
    def _init_model(self):
        """初始化嵌入模型"""
        # 初始化嵌入模型 - 使用Qwen3-Embedding-0.6B模型
        # 首先尝试从本地加载模型，如果不存在则下载
        local_model_path = '/model/Qwen/Qwen3-Embedding-0___6B'
        if not os.path.exists(local_model_path):
            logger.info(f"本地模型不存在，正在下载到 {local_model_path}")
            local_model_path = snapshot_download('Qwen/Qwen3-Embedding-0.6B', cache_dir='/model')
            # state_dict = torch.load(os.path.join(local_model_path, 'pytorch_model.bin'), map_location='cpu')
            # save_file(state_dict, os.path.join(local_model_path, 'model.safetensors'))
        else:
            logger.info(f"使用本地模型: {local_model_path}")
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)
        self.model.to('cpu')
        self.model.eval()
        logger.info(f"读取完成: {local_model_path}")

    def embedding_pipeline(self, texts):
        """处理文本并生成嵌入向量的流水线"""
        logger.info(f"开始embedding_pipeline处理，输入类型: {type(texts)}, 输入数量: {1 if isinstance(texts, str) else len(texts)}")
        
        # 检查输入是否为单个字符串，如果是则转换为列表
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
            logger.debug(f"输入为单个字符串，已转换为列表处理")
        else:
            return_single = False
            logger.debug(f"输入为文本列表，包含 {len(texts)} 个文本")
        
        # 使用tokenizer批量编码文本
        logger.debug(f"开始使用tokenizer编码文本，最大长度: 128")
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        logger.debug(f"文本编码完成，生成token数量: {inputs['input_ids'].shape[1]}")
        
        # 将输入移至CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        logger.debug(f"输入已移至CPU处理")
        
        # 模型推理
        logger.debug(f"开始模型推理生成嵌入向量")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logger.debug(f"模型推理完成")
        
        # 提取嵌入向量（通常使用最后一层隐藏状态的平均值）
        # 不同模型可能有不同的提取方式，这里尝试几种常见方式
        if hasattr(outputs, 'sentence_embedding'):
            embeddings = outputs.sentence_embedding
            logger.debug(f"使用sentence_embedding作为嵌入向量")
        elif hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
            logger.debug(f"使用pooler_output作为嵌入向量")
        elif hasattr(outputs, 'last_hidden_state'):
            # 取最后一层隐藏状态的平均值作为句子嵌入
            embeddings = outputs.last_hidden_state.mean(dim=1)
            logger.debug(f"使用last_hidden_state的平均值作为嵌入向量")
        else:
            # 如果以上都没有，使用模型输出的第一个元素（兜底方案）
            embeddings = list(outputs.values())[0].mean(dim=1)
            logger.debug(f"使用模型输出的第一个元素的平均值作为嵌入向量")
        
        # 转换为numpy数组
        embeddings = embeddings.cpu().numpy()
        logger.debug(f"嵌入向量已转换为numpy数组，形状: {embeddings.shape}")
        
        # 处理返回格式
        results = []
        for embedding in embeddings:
            results.append({'sentence_embedding': embedding.squeeze()})
        logger.debug(f"生成嵌入向量结果 {len(results)} 个")
        
        # 如果输入是单个字符串，返回单个结果
        if return_single and results:
            logger.info(f"embedding_pipeline处理完成，返回单个嵌入向量，维度: {results[0]['sentence_embedding'].shape}")
            return results[0]
        
        logger.info(f"embedding_pipeline处理完成，返回嵌入向量列表，数量: {len(results)}")
        return results
        

    
    def _get_session_store(self, session_id):
        """获取指定session_id的向量存储，如果不存在则创建"""
        if session_id not in self.vector_stores:
            # 创建session的数据库路径
            db_path = os.path.join('vector_db', session_id)
            self.session_paths[session_id] = db_path
            os.makedirs(db_path, exist_ok=True)
            
            # 初始化session的向量存储
            self.vector_stores[session_id] = {
                'vectors': np.array([]),
                'document_chunks': [],
                'document_metadata': []
            }
            
            # 尝试加载已有的向量数据库
            self._load_session_store(session_id)
        
        return self.vector_stores[session_id]
        
    def _load_session_store(self, session_id):
        """加载指定session_id的向量数据库"""
        db_path = self.session_paths[session_id]
        vectors_path = os.path.join(db_path, 'vectors.npy')
        chunks_path = os.path.join(db_path, 'document_chunks.json')
        metadata_path = os.path.join(db_path, 'document_metadata.json')
        
        store = self.vector_stores[session_id]
        
        if os.path.exists(vectors_path) and os.path.exists(chunks_path) and os.path.exists(metadata_path):
            try:
                store['vectors'] = np.load(vectors_path)
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    store['document_chunks'] = json.load(f)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    store['document_metadata'] = json.load(f)
                logger.info(f"成功加载session {session_id}的向量数据库，包含 {len(store['document_chunks'])} 个文档块")
            except Exception as e:
                logger.error(f"加载session {session_id}的向量数据库失败: {str(e)}")
                # 加载失败时创建空数据库
                self._create_empty_database(vectors_path, chunks_path, metadata_path, store)
        else:
            logger.info(f"session {session_id}的向量数据库文件不存在，正在创建新的空数据库: {db_path}")
            # 创建空数据库文件
            self._create_empty_database(vectors_path, chunks_path, metadata_path, store)
    
    def _create_empty_database(self, vectors_path, chunks_path, metadata_path, store):
        """创建空的向量数据库文件"""
        try:
            # 保存空的向量数组
            np.save(vectors_path, np.array([]))
            
            # 保存空的文档块列表
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            
            # 保存空的文档元数据列表
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功创建空的向量数据库文件")
            
            # 从新创建的文件中加载数据
            logger.info(f"正在从新创建的文件中加载向量数据库")
            store['vectors'] = np.load(vectors_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                store['document_chunks'] = json.load(f)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                store['document_metadata'] = json.load(f)
            logger.info(f"成功从新创建的文件中加载向量数据库")
        except Exception as e:
            logger.error(f"创建或加载空向量数据库文件失败: {str(e)}")
            # 加载失败时初始化内存中的数据结构作为后备方案
            store['vectors'] = np.array([])
            store['document_chunks'] = []
            store['document_metadata'] = []
            
    def _save_session_store(self, session_id):
        """保存指定session_id的向量数据库到文件"""
        if session_id not in self.session_paths:
            logger.error(f"保存失败：session {session_id}不存在")
            return False
        
        db_path = self.session_paths[session_id]
        vectors_path = os.path.join(db_path, 'vectors.npy')
        chunks_path = os.path.join(db_path, 'document_chunks.json')
        metadata_path = os.path.join(db_path, 'document_metadata.json')
        
        store = self.vector_stores[session_id]
        
        try:
            if len(store['vectors']) > 0:
                np.save(vectors_path, store['vectors'])
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(store['document_chunks'], f, ensure_ascii=False, indent=2)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(store['document_metadata'], f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存session {session_id}的向量数据库，包含 {len(store['document_chunks'])} 个文档块")
            return True
        except Exception as e:
            logger.error(f"保存session {session_id}的向量数据库失败: {str(e)}")
            return False
    
    def split_text(self, text, chunk_size=500, chunk_overlap=50):
        """将文本分块"""
        logger.info(f"开始文本分块，文本长度: {len(text)}, 块大小: {chunk_size}, 重叠大小: {chunk_overlap}")
        
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
                    
            chunk = text[start:end].strip()
            chunks.append(chunk)
            
            # 确保start向前移动至少一个字符，避免无限循环
            start = max(start + 1, end - chunk_overlap)  # 设置下一个块的起始位置，包含重叠
            
        logger.info(f"文本分块完成，共生成 {len(chunks)} 个块")
        return chunks
    
    def add_document(self, session_id, filename, content):
        """添加文档到指定session_id的向量数据库"""
        logger.info(f"开始添加文档到session {session_id}: {filename}")
        
        # 获取或创建session的向量存储
        store = self._get_session_store(session_id)
        
        # 检查文件是否已经存在
        existing_indices = [i for i, meta in enumerate(store['document_metadata']) if meta['filename'] == filename]
        if existing_indices:
            logger.info(f"检测到文档已存在，将替换旧版本: {filename}")
            # 删除已存在的文档块
            for i in sorted(existing_indices, reverse=True):
                del store['document_chunks'][i]
                del store['document_metadata'][i]
            
            # 重新构建向量存储
            if store['document_chunks']:
                embeddings = [self.embedding_pipeline(chunk)['sentence_embedding'] for chunk in store['document_chunks']]
                store['vectors'] = np.array(embeddings)
            else:
                store['vectors'] = np.array([])
        
        # 分块处理文档内容
        chunks = self.split_text(content)
        if not chunks:
            logger.warning(f"文档分块失败: {filename}")
            return False
        
        # 批量生成文档块的嵌入向量
        batch_results = self.embedding_pipeline(chunks)
        embeddings = [result['sentence_embedding'] for result in batch_results]
        
        # 添加到向量存储
        if len(store['vectors']) == 0:
            store['vectors'] = np.array(embeddings)
        else:
            store['vectors'] = np.vstack([store['vectors'], np.array(embeddings)])
        
        # 保存文档块和元数据
        current_time = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            store['document_chunks'].append(chunk)
            store['document_metadata'].append({
                'filename': filename,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'timestamp': current_time
            })
        
        # 保存数据库
        success = self._save_session_store(session_id)
        
        if success:
            logger.info(f"文档添加成功: {filename}")
        else:
            logger.error(f"文档添加失败: {filename}")
        
        return success
    
    def search(self, session_id, query, top_k=5):
        """搜索指定session_id的向量数据库中与查询最相关的文档块"""
        # 获取session的向量存储
        store = self._get_session_store(session_id)
        
        if len(store['document_chunks']) == 0:
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_pipeline(query)['sentence_embedding']
        
        # 计算查询向量与所有存储向量的L2距离
        distances = np.linalg.norm(store['vectors'] - query_embedding, axis=1)
        
        # 获取距离最小的top_k个索引
        k = min(top_k, len(store['document_chunks']))
        indices = np.argsort(distances)[:k]
        
        # 构建结果列表
        results = []
        for idx in indices:
            results.append({
                'content': store['document_chunks'][idx],
                'metadata': store['document_metadata'][idx],
                'distance': float(distances[idx])
            })
        
        return results
    
    def get_all_documents(self, session_id):
        """获取指定session_id的所有文档的元数据"""
        # 获取session的向量存储
        store = self._get_session_store(session_id)
        
        documents = {}
        for i, meta in enumerate(store['document_metadata']):
            filename = meta['filename']
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'chunk_count': sum(1 for m in store['document_metadata'] if m['filename'] == filename),
                    'upload_time': meta['timestamp']
                }
        
        return list(documents.values())
    
    def delete_document(self, session_id, filename):
        """删除指定session_id中的指定文档"""
        # 获取session的向量存储
        store = self._get_session_store(session_id)
        
        # 找出所有属于该文档的块
        indices_to_remove = [i for i, meta in enumerate(store['document_metadata']) if meta['filename'] == filename]
        
        if not indices_to_remove:
            return False
        
        # 删除文档块和元数据
        for i in sorted(indices_to_remove, reverse=True):
            del store['document_chunks'][i]
            del store['document_metadata'][i]
        
        # 重新构建向量存储
        if store['document_chunks']:
            embeddings = [self.embedding_pipeline(chunk)['sentence_embedding'] for chunk in store['document_chunks']]
            store['vectors'] = np.array(embeddings)
        else:
            store['vectors'] = np.array([])
        
        # 保存数据库
        success = self._save_session_store(session_id)
        
        return success