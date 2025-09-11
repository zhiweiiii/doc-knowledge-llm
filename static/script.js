document.addEventListener('DOMContentLoaded', function() {
    // 元素引用
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadForm = document.getElementById('uploadForm');
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    let currentFileName = null;
    let isWaitingForResponse = false;
    
    // 文件上传相关事件
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });
    
    // 拖放功能
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        }, false);
    });
    
    dropArea.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) {
            uploadFile(file);
        }
    }, false);
    
    // 上传文件函数
    function uploadFile(file) {
        // 检查文件类型
        const validTypes = ['.pdf', '.doc', '.docx', '.txt'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(fileExtension)) {
            uploadStatus.textContent = '不支持的文件类型，请上传PDF、Word或TXT格式的知识库文件';
            uploadStatus.className = 'upload-status error';
            return;
        }
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('file', file);
        
        // 显示上传中状态
        uploadStatus.textContent = '上传中...';
        uploadStatus.className = 'upload-status';
        
        // 发送上传请求
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadStatus.textContent = `知识库文件 ${data.filename} 上传成功！`;
                uploadStatus.className = 'upload-status success';
                currentFileName = data.filename;
                
                // 启用聊天功能
                messageInput.disabled = false;
                sendButton.disabled = false;
                
                // 添加系统消息
                addMessage('system', `知识库文件 ${data.filename} 已上传，您可以开始提问了。`);
            } else {
                uploadStatus.textContent = data.error || '上传失败';
                uploadStatus.className = 'upload-status error';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadStatus.textContent = '上传过程中发生错误';
            uploadStatus.className = 'upload-status error';
        });
    }
    
    // 聊天相关功能
    sendButton.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = messageInput.value.trim();
        if (message && !isWaitingForResponse) {
            // 清空输入框
            messageInput.value = '';
            
            // 添加用户消息到聊天区域
            addMessage('user', message);
            
            // 显示加载指示器
            showTypingIndicator();
            
            // 设置等待状态
            isWaitingForResponse = true;
            messageInput.disabled = true;
            sendButton.disabled = true;
            
            // 发送消息到服务器
            fetchChatResponse(message);
        }
    }
    
    function addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // 滚动到底部
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typingIndicator';
        
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'typing-indicator';
        indicatorDiv.textContent = '正在思考...';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicatorDiv.appendChild(dot);
        }
        
        typingDiv.appendChild(indicatorDiv);
        chatMessages.appendChild(typingDiv);
        
        // 滚动到底部
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // 清理文本内容，确保省略多余格式和前缀
    function cleanTextForDisplay(text) {
        if (!text) return text;
        
        let cleanedText = text;
        
        // 移除常见的前缀格式
        cleanedText = cleanedText.replace(/^assistant:/i, '').trim();
        cleanedText = cleanedText.replace(/^\s*回答:/i, '').trim();
        cleanedText = cleanedText.replace(/^\s*AI:/i, '').trim();
        cleanedText = cleanedText.replace(/^\s*回复:/i, '').trim();
        cleanedText = cleanedText.replace(/^\s*user:/i, '').trim();
        cleanedText = cleanedText.replace(/^\s*user/i, '').trim();
        
<<<<<<< HEAD
        // 处理可能的转义字符，但保留原始字符编码
        cleanedText = cleanedText.replace(/\\n/g, ' ');
        cleanedText = cleanedText.replace(/\\t/g, ' ');
        
        // 移除多余的空格，但保留Unicode空格
=======
        // 处理可能的转义字符
        cleanedText = cleanedText.replace(/\\n/g, ' ');
        cleanedText = cleanedText.replace(/\\t/g, ' ');
        
        // 移除多余的空格
>>>>>>> ae49bb1cb06bf6104abd6d1db5c73b7bd67a7d14
        cleanedText = cleanedText.replace(/\s+/g, ' ').trim();
        
        return cleanedText;
    }
    
    function fetchChatResponse(message) {
        // 构建包含文件名的查询参数
        const queryParams = new URLSearchParams({
            text: message,
            file: currentFileName || ''
        });
        
        // 创建EventSource对象来接收服务器发送的事件
        const eventSource = new EventSource(`/chat?${queryParams}`);
        
        let responseText = '';
        let messageDiv = null;
        
        eventSource.onmessage = function(event) {
            const message = event.data.trim();
            
        
            
            // 检查是否是错误消息
            if (message.startsWith('发生错误:')) {
                if (!messageDiv) {
                    removeTypingIndicator();
                    const thinkingMessage = document.getElementById('thinkingMessage');
                    if (thinkingMessage) {
                        thinkingMessage.remove();
                    }
                    
                    messageDiv = document.createElement('div');
                    messageDiv.className = 'message system error';
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';
                    messageDiv.appendChild(contentDiv);
                    
                    chatMessages.appendChild(messageDiv);
                }
                
                responseText = message;
                
                const contentDiv = messageDiv.querySelector('.message-content');
                contentDiv.textContent = responseText;
                
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else {
                // 正常内容消息
                if (!messageDiv) {
<<<<<<< HEAD
=======
                    // 移除思考提示
                    removeTypingIndicator();
>>>>>>> ae49bb1cb06bf6104abd6d1db5c73b7bd67a7d14
                    const thinkingMessage = document.getElementById('thinkingMessage');
                    if (thinkingMessage) {
                        thinkingMessage.remove();
                    }
                    
                    // 创建新的消息div
                    messageDiv = document.createElement('div');
                    messageDiv.className = 'message assistant';
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';
                    messageDiv.appendChild(contentDiv);
                    
                    chatMessages.appendChild(messageDiv);
                }
                
                // 追加新的文本内容
                responseText += message;
                
                // 处理文本内容，确保省略多余格式和前缀
                let displayText = responseText;
                
                // 清除任何可能的格式标记或前缀
                displayText = cleanTextForDisplay(displayText);
<<<<<<< HEAD
                if(displayText){
                    // 移除思考提示
                    removeTypingIndicator();
                }
                // 更新消息内容，使用textContent确保正确显示所有字符
=======
                
                // 更新消息内容
>>>>>>> ae49bb1cb06bf6104abd6d1db5c73b7bd67a7d14
                const contentDiv = messageDiv.querySelector('.message-content');
                contentDiv.textContent = displayText;
                
                // 滚动到底部
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        };
        
        eventSource.onerror = function() {
            // 关闭事件源
            eventSource.close();
            
            // 如果没有收到任何消息，显示错误
            if (!responseText) {
                removeTypingIndicator();
                addMessage('system', '无法获取回复，请重试。');
            }
            
            // 重置状态
            isWaitingForResponse = false;
            messageInput.disabled = false;
            sendButton.disabled = false;
        };
    }
});