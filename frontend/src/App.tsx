import React, { useState, useRef } from 'react'
import { Send, Paperclip, Settings, Plus, MessageSquare, Menu, FileText, CheckCircle } from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  file?: File
  documentId?: string
  documentName?: string
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  documentContext?: {
    documentId: string
    documentName: string
    documentContent: string
    uploadedAt: Date
  }
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState('tinyllama')
  const [showSettings, setShowSettings] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const activeConversation = conversations.find(c => c.id === activeConversationId)

  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date()
    }
    setConversations(prev => [newConversation, ...prev])
    setActiveConversationId(newConversation.id)
  }

  const sendMessage = async (content: string, file?: File) => {
    let fileContent = '';
    if (file) {
      try {
        fileContent = await file.text();
      } catch (error) {
        console.error('Failed to read file content:', error);
      }
    }
    if (!content.trim() && !file) return

    let currentConversationId = activeConversationId

    if (!currentConversationId) {
      createNewConversation()
      currentConversationId = conversations[0]?.id || Date.now().toString()
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
      file,
      documentId: file ? undefined : activeConversation?.documentContext?.documentId,
      documentName: file ? file.name : activeConversation?.documentContext?.documentName
    }

    setConversations(prev => prev.map(conv => 
      conv.id === currentConversationId 
        ? { ...conv, messages: [...conv.messages, userMessage] }
        : conv
    ))

    setIsLoading(true)
    setInputMessage('')

    try {
      // Always use upload endpoint if there's a file OR existing document context
      const hasDocumentContext = !!activeConversation?.documentContext
      const shouldUseUploadEndpoint = file || hasDocumentContext

      let requestInit: RequestInit

      if (shouldUseUploadEndpoint) {
        // Use multipart form data for upload endpoint
        const formData = new FormData()
        
        if (file) {
          // New file upload
          formData.append('file', file)
          formData.append('prompt', content || 'Analyze this document')
        } else if (hasDocumentContext && activeConversation?.documentContext) {
          // Create a temporary file from existing document context
          const documentBlob = new Blob([activeConversation.documentContext.documentContent], { type: 'text/plain' })
          const documentFile = new File([documentBlob], activeConversation.documentContext.documentName, { type: 'text/plain' })
          formData.append('file', documentFile)
          formData.append('prompt', content)
        }
        
        formData.append('model', selectedModel)
        formData.append('max_tokens', '100')
        formData.append('auto_process_document', 'true')
        formData.append('use_document_context', 'true')
        formData.append('use_memory', hasDocumentContext ? 'true' : 'false')
        formData.append('session_id', currentConversationId || '')
        
        requestInit = { method: 'POST', body: formData }
      } else {
        // Regular JSON request for simple text generation
        requestInit = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: content,
            model: selectedModel,
            max_tokens: 100,
            temperature: 0.7,
            use_memory: true,  // Enable memory for conversation context
            session_id: currentConversationId,  // Include session ID
            auto_process_document: false,
            use_document_context: false
          })
        }
      }

      const endpoint = shouldUseUploadEndpoint ? '/api/v1/generate/upload' : '/api/v1/generate'

      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000'
      const response = await fetch(`${apiBaseUrl}${endpoint}`, requestInit)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || data.text || 'No response received',
        timestamp: new Date(),
        documentId: data.document_id,
        documentName: file?.name
      }

      setConversations(prev => prev.map(conv => 
        conv.id === currentConversationId 
          ? { 
              ...conv, 
              messages: [...conv.messages, assistantMessage],
              title: conv.messages.length === 1 ? content.slice(0, 30) + '...' : conv.title,
              // Set document context if file was uploaded
              documentContext: file ? {
                documentId: data.document_id || Date.now().toString(),
                documentName: file.name,
                documentContent: fileContent,
                uploadedAt: new Date()
              } : conv.documentContext
            }
          : conv
      ))

    } catch (error) {
      console.error('Error sending message:', error)
      console.error('Error details:', error instanceof Error ? error.message : String(error))
      
      let errorContent = 'Failed to send message'
      if (error instanceof Error) {
        if (error.message.includes('Failed to fetch')) {
          errorContent = 'Cannot connect to AI server. Please make sure the backend server is running on http://localhost:3000'
        } else if (error.message.includes('HTTP error')) {
          errorContent = `Server error: ${error.message}. Check browser console for details.`
        } else {
          errorContent = error.message
        }
      }
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${errorContent}`,
        timestamp: new Date()
      }

      setConversations(prev => prev.map(conv => 
        conv.id === currentConversationId 
          ? { ...conv, messages: [...conv.messages, errorMessage] }
          : conv
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(inputMessage)
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      sendMessage(inputMessage || 'Analyze this document', file)
    }
  }

  return (
    <div className="flex h-screen bg-white dark:bg-gpt-gray-900">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`sidebar w-64 h-full flex flex-col md:relative md:translate-x-0 ${sidebarOpen ? 'open' : ''}`}>
        {/* Header */}
        <div className="p-4 border-b border-gpt-gray-200 dark:border-gpt-gray-700">
          <button 
            onClick={createNewConversation}
            className="w-full btn-primary flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* Conversations */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
          {conversations.map(conversation => (
            <button
              key={conversation.id}
              onClick={() => setActiveConversationId(conversation.id)}
              className={`w-full text-left p-3 rounded-lg mb-2 transition-colors duration-200 ${
                activeConversationId === conversation.id
                  ? 'bg-gpt-gray-200 dark:bg-gpt-gray-700'
                  : 'hover:bg-gpt-gray-100 dark:hover:bg-gpt-gray-800'
              }`}
            >
              <div className="flex items-center gap-2">
                {conversation.documentContext ? (
                  <FileText className="w-4 h-4 text-gpt-green-600" />
                ) : (
                  <MessageSquare className="w-4 h-4 text-gpt-gray-500" />
                )}
                <span className="text-sm font-medium truncate">{conversation.title}</span>
              </div>
              <div className="text-xs text-gpt-gray-500 mt-1">
                {conversation.documentContext && (
                  <div className="text-gpt-green-600 text-xs mb-1">
                    📄 {conversation.documentContext.documentName}
                  </div>
                )}
                {conversation.createdAt.toLocaleDateString()}
              </div>
            </button>
          ))}
        </div>

        {/* Model Selection & Settings */}
        <div className="p-4 border-t border-gpt-gray-200 dark:border-gpt-gray-700">
          <div className="mb-3">
            <label className="block text-xs font-medium text-gpt-gray-600 dark:text-gpt-gray-400 mb-1">
              Model
            </label>
            <select 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full chat-input text-sm"
            >
              <option value="tinyllama">TinyLlama (Recommended)</option>
              <option value="llama-generic">Llama Generic</option>
            </select>
          </div>
          
          <button 
            onClick={() => setShowSettings(!showSettings)}
            className="btn-secondary w-full flex items-center justify-center gap-2"
          >
            <Settings className="w-4 h-4" />
            Settings
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Mobile header */}
        <div className="md:hidden flex items-center justify-between p-4 border-b border-gpt-gray-200 dark:border-gpt-gray-700">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-md hover:bg-gpt-gray-100 dark:hover:bg-gpt-gray-700 transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          <h1 className="text-lg font-semibold">AI Inference Server</h1>
          <div className="w-9" /> {/* Spacer */}
        </div>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          {/* Document Context Indicator */}
          {activeConversation?.documentContext && (
            <div className="max-w-4xl mx-auto mb-4">
              <div className="bg-gpt-green-50 dark:bg-gpt-green-900/20 border border-gpt-green-200 dark:border-gpt-green-600 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-gpt-green-600" />
                  <span className="text-sm font-medium text-gpt-green-800 dark:text-gpt-green-200">
                    Document Context Active
                  </span>
                  <CheckCircle className="w-4 h-4 text-gpt-green-600" />
                </div>
                <div className="text-xs text-gpt-green-700 dark:text-gpt-green-300 mt-1">
                  📄 {activeConversation.documentContext.documentName} - You can now ask questions about this document
                </div>
              </div>
            </div>
          )}
          
          {activeConversation?.messages.length === 0 || !activeConversation ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gpt-gray-500">
                <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
                <p>Choose a model and send a message to get started</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto">
              {activeConversation.messages.map(message => (
                <div key={message.id} className="animate-fade-in">
                  <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={message.role === 'user' ? 'message-user' : 'message-assistant'}>
                      {message.file && (
                        <div className="flex items-center gap-2 mb-2 text-sm text-gpt-gray-600 dark:text-gpt-gray-400">
                          <Paperclip className="w-4 h-4" />
                          {message.file.name}
                        </div>
                      )}
                      {message.documentName && !message.file && (
                        <div className="flex items-center gap-2 mb-2 text-sm text-gpt-gray-600 dark:text-gpt-gray-400">
                          <FileText className="w-4 h-4" />
                          <span className="italic">Referencing: {message.documentName}</span>
                        </div>
                      )}
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      <div className="text-xs text-gpt-gray-500 mt-2">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start animate-fade-in">
                  <div className="message-assistant">
                    <div className="typing-dots">
                      <div className="typing-dot" style={{'--delay': '0s'} as React.CSSProperties}></div>
                      <div className="typing-dot" style={{'--delay': '0.2s'} as React.CSSProperties}></div>
                      <div className="typing-dot" style={{'--delay': '0.4s'} as React.CSSProperties}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gpt-gray-200 dark:border-gpt-gray-700 p-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="relative flex items-end gap-2">
              <div className="flex-1 relative">
                <div
                  className="chat-input w-full min-h-[44px] max-h-[200px] overflow-y-auto border-2 border-dashed border-gpt-gray-300 dark:border-gpt-gray-600 rounded-lg p-3 transition-colors duration-200 hover:border-gpt-green-500 focus-within:border-gpt-green-500"
                  onDragOver={(e) => {
                    e.preventDefault()
                    e.currentTarget.classList.add('border-gpt-green-500', 'bg-gpt-green-50', 'dark:bg-gpt-green-900/20')
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault()
                    e.currentTarget.classList.remove('border-gpt-green-500', 'bg-gpt-green-50', 'dark:bg-gpt-green-900/20')
                  }}
                  onDrop={(e) => {
                    e.preventDefault()
                    e.currentTarget.classList.remove('border-gpt-green-500', 'bg-gpt-green-50', 'dark:bg-gpt-green-900/20')
                    const file = e.dataTransfer.files[0]
                    if (file) {
                      sendMessage(inputMessage || 'Analyze this document', file)
                    }
                  }}
                >
                  <textarea
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder={
                      activeConversation?.documentContext 
                        ? `Ask questions about ${activeConversation.documentContext.documentName}...`
                        : "Message AI Inference Server... (or drag & drop a file)"
                    }
                    className="w-full resize-none bg-transparent border-none outline-none pr-12"
                    rows={1}
                    style={{
                      height: Math.min(150, Math.max(20, inputMessage.split('\n').length * 24))
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSubmit(e)
                      }
                    }}
                  />
                  
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute right-12 top-1/2 transform -translate-y-1/2 p-1.5 rounded-md hover:bg-gpt-gray-100 dark:hover:bg-gpt-gray-700 transition-colors"
                    title="Upload file"
                  >
                    <Paperclip className="w-4 h-4 text-gpt-gray-500" />
                  </button>
                  
                  <button
                    type="submit"
                    disabled={(!inputMessage.trim() && !fileInputRef.current?.files?.[0]) || isLoading}
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1.5 rounded-md bg-gpt-green-500 hover:bg-gpt-green-600 disabled:bg-gpt-gray-300 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="w-4 h-4 text-white" />
                  </button>
                </div>
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileUpload}
                accept=".txt,.md,.pdf,.docx"
                className="hidden"
              />
            </div>
            
            <div className="text-xs text-gpt-gray-500 mt-2 text-center">
              AI can make mistakes. Verify important information.
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
