import logging
import os
import uuid
import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory, session, Response
import PyPDF2
import docx

from QwenThread import QwenThread
from VectorDB import VectorDatabase

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'doc_knowledge_llm_secret_key'  # 设置密钥，用于会话加密

# 设置Flask应用的日志级别为DEBUG
app.logger.setLevel(logging.DEBUG)

# 创建上传文件的目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 创建会话文件存储目录
SESSION_FILES_FOLDER = 'session_files'
os.makedirs(SESSION_FILES_FOLDER, exist_ok=True)

# 创建向量数据库存储目录
VECTOR_DB_FOLDER = 'vector_db'
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# 生成会话ID
def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

# 在应用启动时就初始化全局共用的VectorDatabase实例
app.logger.info(f"应用启动，开始初始化全局共用的向量数据库")
vector_db = VectorDatabase()
app.logger.info(f"全局共用向量数据库初始化完成")

# 获取会话的向量数据库
def get_vector_db():
    app.logger.info(f"直接调用已初始化的全局共用向量数据库")
    return vector_db

# 获取会话文件路径
def get_session_file_path(filename):
    session_id = get_session_id()
    session_dir = os.path.join(SESSION_FILES_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return os.path.join(session_dir, filename)

# 初始化QwenThread
def init_qwen_thread():
    global qwenThread
    qwenThread = QwenThread()

# 定义路由和视图函数
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/message', methods=['GET'])
def chat():
    app.logger.info("开始")
    ### 使用url
    text = request.values.get('text')
    if text is None:
        return "请输入信息"
    
    # 使用向量数据库检索相关文档内容
    relevant_content = ""
    if 'file_mappings' in session and len(session['file_mappings']) > 0:
        try:
            # 获取向量数据库
            
            # 搜索相关文档块
            session_id = get_session_id()
            search_results = vector_db.search(session_id, text, top_k=5)
            
            if search_results:
                # 构建相关内容
                for i, result in enumerate(search_results):
                    relevant_content += result['content'] + "\n\n"
        except Exception as e:
            app.logger.error(f"向量数据库检索错误: {str(e)}")
            # 即使检索失败，也继续处理用户问题，只是没有知识库支持
    
    # 使用SSE（Server-Sent Events）实现流式响应，返回纯文本格式
    # 在请求上下文有效时获取用户ID
    user_id = get_session_id()
    
    def generate():
        try:  
            # 然后发送实际的流式响应，直接返回纯文本内容
            # 确保中文和特殊字符正确编码
            # 使用新方法，将知识库内容和问题分开传递
            for chunk in qwenThread.stream_chat_with_knowledge(text, relevant_content, user_id=user_id):
                # 确保内容是字符串并正确编码
                chunk_str = str(chunk) if chunk else ''
                yield f"data: {chunk_str}\n\n"
        except Exception as e:
            app.logger.error(f"流式响应错误: {str(e)}")
            # 返回错误消息
            error_str = str(e) if e else '未知错误'
            yield f"data: 发生错误: {error_str}\n\n"
    app.logger.info("结束")
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存原始文件
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # 解析文件内容并保存到会话专属文件中
    try:
        file_content = ""
        if file.filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        elif file.filename.lower().endswith('.pdf'):
            # 处理PDF文件
            pdf_reader = PyPDF2.PdfReader(file_path)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                file_content += page.extract_text() + "\n"
        elif file.filename.lower().endswith(('.doc', '.docx')):
            # 处理Word文件
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                file_content += para.text + "\n"
        
        # 将文件内容保存到会话专属文件中
        if file_content:
            # 生成会话文件名（使用原始文件名）
            session_file_path = get_session_file_path(f"{file.filename}.content")
            
            # 写入文件内容到会话专属文件
            with open(session_file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            # 在会话中只保存文件名的映射关系
            if 'file_mappings' not in session:
                session['file_mappings'] = {}
            
            # 记录文件映射信息
            session['file_mappings'][file.filename] = {
                'session_file': f"{file.filename}.content",
                'original_file': file.filename,
                'upload_time': datetime.now().isoformat()
            }
            session.modified = True
            
            app.logger.info(f"文件内容已保存到会话专属文件: {session_file_path}")
            
            # 创建一个用于流式传输进度的响应，使用队列和线程
            from queue import Queue
            import threading
            
            # 创建一个队列用于在处理线程和响应生成器之间传递消息
            message_queue = Queue()
            session_id = get_session_id()
            # 进度回调函数
            def progress_callback(progress, current_batch, total_batches):
                # 构建进度信息
                progress_info = {
                    'progress': progress,
                    'current_batch': current_batch,
                    'total_batches': total_batches,
                    'status': 'processing'
                }
                # 将进度信息放入队列
                message_queue.put(('progress', progress_info))
             
            # 文档处理线程函数
            def process_document():
                try:
                    app.logger.info(f"添加文件: {file.filename}")
                
                    # 尝试添加文档到向量数据库，捕获所有错误
                    success = vector_db.add_document(session_id, file.filename, file_content, progress_callback=progress_callback)
                    
                    if success:
                        app.logger.info(f"文件已添加到向量数据库: {file.filename}")
                        # 发送完成信息
                        finish_info = {
                            'progress': 100,
                            'status': 'completed',
                            'filename': file.filename
                        }
                        message_queue.put(('completed', finish_info))
                    else:
                        # 从日志中提取最后一个相关错误信息
                        error_message = '文件添加失败'
                        # 尝试从最近的日志中获取更具体的错误信息
                        app.logger.error(f"文件添加失败: {file.filename}")
                        # 发送错误信息
                        error_info = {
                            'progress': 100,
                            'status': 'error',
                            'message': error_message
                        }
                        message_queue.put(('error', error_info))
                except Exception as e:
                    app.logger.error(f"添加文件到向量数据库失败: {str(e)}")
                    # 发送错误信息，包含具体错误原因
                    error_info = {
                        'progress': 100,
                        'status': 'error',
                        'message': f'添加文件失败: {str(e)}'
                    }
                    message_queue.put(('error', error_info))
                finally:
                    # 标记处理完成
                    message_queue.put(('done', None))
            
            # 创建并启动处理线程
            processing_thread = threading.Thread(target=process_document)
            processing_thread.daemon = True
            processing_thread.start()
            
            # 生成器函数，从队列中获取消息并发送给客户端
            def generate():
                while True:
                    # 从队列获取消息
                    message_type, message_data = message_queue.get()
                    
                    # 如果处理完成，退出循环
                    if message_type == 'done':
                        break
                    
                    # 发送消息给客户端
                    if message_data:
                        yield f"data: {json.dumps(message_data)}\n"
            
            # 使用SSE流式响应
            response = Response(generate(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            return response
        else:
            return jsonify({'success': False, 'error': '文件内容为空'}), 400
    except Exception as e:
        app.logger.error(f"解析文件错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # 初始化QwenThread
    init_qwen_thread()
    app.run(host="0.0.0.0", port=80)

