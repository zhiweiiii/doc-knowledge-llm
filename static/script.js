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
    
    let uploadedFiles = []; // 存储所有上传的文件名
    let isWaitingForResponse = false;
    
    // 文件上传相关事件
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });
    
    // 点击上传区域任意位置也能触发文件选择
    dropArea.addEventListener('click', (e) => {
        // 确保点击的是上传区域本身或其子元素（除了已经处理过的按钮）
        if (e.target === dropArea || e.target.closest('.upload-placeholder')) {
            fileInput.click();
        }
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            // 处理所有选中的文件
            for (let i = 0; i < fileInput.files.length; i++) {
                uploadFile(fileInput.files[i]);
            }
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
        // 处理所有拖放的文件
        for (let i = 0; i < e.dataTransfer.files.length; i++) {
            uploadFile(e.dataTransfer.files[i]);
        }
    }, false);
    
    // 上传文件函数
    function uploadFile(file) {
        // 检查文件类型
        const validTypes = ['.pdf', '.doc', '.docx', '.txt'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(fileExtension)) {
            uploadStatus.textContent = `不支持的文件类型: ${file.name}`;
            uploadStatus.className = 'upload-status error';
            setTimeout(() => {
                uploadStatus.textContent = '';
            }, 3000);
            return;
        }
        
        // 检查文件是否已经上传过
        if (uploadedFiles.includes(file.name)) {
            uploadStatus.textContent = `文件 ${file.name} 已经上传过了`;
            uploadStatus.className = 'upload-status error';
            setTimeout(() => {
                uploadStatus.textContent = '';
            }, 3000);
            return;
        }
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('file', file);
        
        // 显示上传中状态
        uploadStatus.textContent = `上传中: ${file.name}`;
        uploadStatus.className = 'upload-status';
        
        // 发送上传请求
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 添加到已上传文件列表
                uploadedFiles.push(data.filename);
                
                uploadStatus.textContent = `知识库文件 ${data.filename} 上传成功！`;
                uploadStatus.className = 'upload-status success';
                
                // 启用聊天功能
                messageInput.disabled = false;
                sendButton.disabled = false;
                
                // 添加系统消息
                const message = uploadedFiles.length === 1 
                    ? `知识库文件 ${data.filename} 已上传，您可以开始提问了。` 
                    : `知识库文件 ${data.filename} 已上传，当前共上传了 ${uploadedFiles.length} 个文件。`;
                addMessage('system', message);
                
                // 3秒后清除上传状态
                setTimeout(() => {
                    uploadStatus.textContent = '';
                }, 3000);
            } else {
                uploadStatus.textContent = `上传失败: ${data.error || '未知错误'}`;
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
        
        // 处理可能的转义字符，但保留原始字符编码
        cleanedText = cleanedText.replace(/\\n/g, ' ');
        cleanedText = cleanedText.replace(/\\t/g, ' ');
        
        // 移除多余的空格，但保留Unicode空格
        cleanedText = cleanedText.replace(/\s+/g, ' ').trim();
        
        return cleanedText;
    }
    
    function fetchChatResponse(message) {
        // 只传递用户消息，文件信息由服务器从会话中获取
        const queryParams = new URLSearchParams({
            text: message
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
                if(displayText){
                    // 移除思考提示
                    removeTypingIndicator();
                }
                // 更新消息内容，使用textContent确保正确显示所有字符
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