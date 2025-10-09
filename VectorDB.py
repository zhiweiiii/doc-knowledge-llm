

import json
import numpy as np
from modelscope.hub.snapshot_download import snapshot_download
from datetime import datetime
from safetensors.torch import save_file
import torch
import os
import logging
from transformers import AutoTokenizer, AutoModel
import faiss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        """初始化向量数据库"""
        self.session_paths = {}
        self.vector_stores = {}
        self._init_model()
        
    def _init_model(self):
        """初始化嵌入模型"""
        model_name = 'iic/nlp_gte_sentence-embedding_chinese-large'
        local_model_path = '/model/iic/nlp_gte_sentence-embedding_chinese-large'
        
        if not os.path.exists(local_model_path):
            logger.info(f"下载模型到 {local_model_path}")
            local_model_path = snapshot_download(model_name, cache_dir='/model')
            state_dict = torch.load(os.path.join(local_model_path, 'pytorch_model.bin'), map_location='cpu')
            save_file(state_dict, os.path.join(local_model_path, 'model.safetensors'))
        
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)
        self.model.to('cpu')
        self.model.eval()
        logger.info(f"模型加载完成: {local_model_path}")

    def embedding_pipeline(self, texts, batch_size=128, progress_callback=None):
        """处理文本并生成嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
            
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            # 进度回调
            if progress_callback:
                try:
                    progress_callback((current_batch / total_batches) * 100, current_batch, total_batches)
                except Exception as e:
                    logger.warning(f"进度回调错误: {e}")
            
            # 模型推理
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 提取嵌入向量
            if hasattr(outputs, 'sentence_embedding'):
                embeddings = outputs.sentence_embedding
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = list(outputs.values())[0].mean(dim=1)
            
            all_embeddings.extend(embeddings.cpu().numpy())
            
            # 清理内存
            del inputs, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 处理完成回调
        if progress_callback:
            try:
                progress_callback(100, total_batches, total_batches)
            except Exception as e:
                logger.warning(f"完成回调错误: {e}")
        
        # 格式化结果
        results = [{'sentence_embedding': emb.squeeze()} for emb in all_embeddings]
        return results[0] if return_single and results else results
    
    def _get_session_store(self, session_id):
        """获取指定session_id的向量存储"""
        if session_id not in self.vector_stores:
            db_path = os.path.join('vector_db', session_id)
            self.session_paths[session_id] = db_path
            os.makedirs(db_path, exist_ok=True)
            
            self.vector_stores[session_id] = {
                'index': None,
                'document_chunks': [],
                'document_metadata': []
            }
            
            self._load_session_store(session_id)
        
        return self.vector_stores[session_id]
        
    def _load_session_store(self, session_id):
        """加载向量数据库"""
        db_path = self.session_paths[session_id]
        index_path = os.path.join(db_path, 'faiss_index')
        chunks_path = os.path.join(db_path, 'document_chunks.json')
        metadata_path = os.path.join(db_path, 'document_metadata.json')
        
        store = self.vector_stores[session_id]
        
        if all(os.path.exists(path) for path in [index_path, chunks_path, metadata_path]):
            try:
                store['index'] = faiss.read_index(index_path)
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    store['document_chunks'] = json.load(f)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    store['document_metadata'] = json.load(f)
                logger.info(f"加载数据库完成，包含 {len(store['document_chunks'])} 个文档块")
            except Exception as e:
                logger.error(f"加载数据库失败: {str(e)}")
                self._create_empty_database(index_path, chunks_path, metadata_path, store)
        else:
            logger.info(f"创建新的空数据库")
            self._create_empty_database(index_path, chunks_path, metadata_path, store)
    
    def _create_empty_database(self, index_path, chunks_path, metadata_path, store):
        """创建空的向量数据库"""
        try:
            # 先使用一个示例文本测试模型输出的维度
            test_embedding = self.embedding_pipeline("测试文本")['sentence_embedding']
            dimension = test_embedding.shape[0]
            logger.info(f"模型输出维度检测: {dimension}")
            
            store['index'] = faiss.IndexFlatL2(dimension)
            faiss.write_index(store['index'], index_path)
            
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
                
            # 直接使用创建的空数据，跳过重新加载
            store['document_chunks'] = []
            store['document_metadata'] = []
        except Exception as e:
            logger.error(f"创建空数据库失败: {str(e)}")
            # 出现错误时默认使用1024维（iic/nlp_gte_sentence-embedding_chinese-large的标准维度）
            store['index'] = faiss.IndexFlatL2(1024)
            store['document_chunks'] = []
            store['document_metadata'] = []
    
    def _save_session_store(self, session_id):
        """保存向量数据库"""
        if session_id not in self.session_paths:
            logger.error(f"保存失败：session不存在")
            return False
        
        db_path = self.session_paths[session_id]
        index_path = os.path.join(db_path, 'faiss_index')
        chunks_path = os.path.join(db_path, 'document_chunks.json')
        metadata_path = os.path.join(db_path, 'document_metadata.json')
        
        store = self.vector_stores[session_id]
        
        try:
            if store['index'] is not None and store['index'].ntotal > 0:
                faiss.write_index(store['index'], index_path)
            
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(store['document_chunks'], f, ensure_ascii=False, indent=2)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(store['document_metadata'], f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存数据库完成，包含 {len(store['document_chunks'])} 个文档块")
            return True
        except Exception as e:
            logger.error(f"保存数据库失败: {str(e)}")
            return False
    
    def split_text(self, text, chunk_size=500, chunk_overlap=50):
        """将文本分块"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # 尝试在句子边界处分割
            if end < text_length:
                punctuation_positions = [text.rfind(p, start, end) for p in ['.', '?', '!', '\n']]
                punctuation_positions = [pos for pos in punctuation_positions if pos > start + chunk_size * 0.5]
                
                if punctuation_positions:
                    end = max(punctuation_positions) + 1
            
            chunks.append(text[start:end].strip())
            start = max(start + 1, end - chunk_overlap)
        
        return chunks
    
    def add_document(self, session_id, filename, content, progress_callback=None):
        """添加文档到向量数据库"""
        logger.info(f"添加文档: {filename}")
        
        try:
            store = self._get_session_store(session_id)
            chunks = self.split_text(content)
            
            if not chunks:
                logger.warning(f"文档分块失败")
                return False
            
            # 生成嵌入向量
            batch_results = self.embedding_pipeline(chunks, progress_callback=progress_callback)
            embeddings = np.array([result['sentence_embedding'] for result in batch_results])
            
            # 添加到向量存储
            if store['index'] is None:
                dimension = embeddings.shape[1]
                store['index'] = faiss.IndexFlatL2(dimension)
            elif store['index'].d != embeddings.shape[1]:
                raise ValueError(f"索引维度不匹配: 期望{store['index'].d}，实际{embeddings.shape[1]}")
            
            store['index'].add(embeddings)
            
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
            
            return self._save_session_store(session_id)
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    def search(self, session_id, query, top_k=5):
        """搜索相关文档块"""
        store = self._get_session_store(session_id)
        
        if not store['document_chunks']:
            return []
        
        # 生成查询向量并搜索
        query_embedding = self.embedding_pipeline(query)['sentence_embedding']
        distances, indices = store['index'].search(np.expand_dims(query_embedding, axis=0), top_k)
        
        # 构建结果
        return [{
            'content': store['document_chunks'][indices[0][i]],
            'metadata': store['document_metadata'][indices[0][i]],
            'distance': float(distances[0][i])
        } for i in range(len(indices[0]))]
    
    def get_all_documents(self, session_id):
        """获取所有文档元数据"""
        store = self._get_session_store(session_id)
        documents = {}
        
        for meta in store['document_metadata']:
            filename = meta['filename']
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'chunk_count': sum(1 for m in store['document_metadata'] if m['filename'] == filename),
                    'upload_time': meta['timestamp']
                }
        
        return list(documents.values())
    
    def delete_document(self, session_id, filename):
        """删除指定文档"""
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
            embeddings = np.array([self.embedding_pipeline(chunk)['sentence_embedding'] for chunk in store['document_chunks']])
            dimension = embeddings.shape[1]
            store['index'] = faiss.IndexFlatL2(dimension)
            store['index'].add(embeddings)
        else:
            # 当没有文档时，使用模型的实际输出维度创建空索引
            try:
                test_embedding = self.embedding_pipeline("测试文本")['sentence_embedding']
                dimension = test_embedding.shape[0]
            except Exception as e:
                logger.warning(f"检测模型维度失败，使用默认值1024: {str(e)}")
                dimension = 1024
            store['index'] = faiss.IndexFlatL2(dimension)
        
        # 保存数据库
        return self._save_session_store(session_id)