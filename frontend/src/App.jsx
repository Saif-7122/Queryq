import { useState, useRef, useCallback, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ─── LoadingSkeleton ────────────────────────────────────────────────────────
function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-2 w-full animate-pulse">
      {[1, 2].map(i => (
        <div key={i} className="flex items-center gap-3 bg-white border border-slate-200 rounded-xl px-3 py-3 shadow-sm">
          <div className="w-8 h-8 rounded-lg bg-slate-200 shrink-0" />
          <div className="flex-1 min-w-0 flex flex-col gap-2">
            <div className="h-3.5 bg-slate-200 rounded w-3/4" />
            <div className="h-2.5 bg-slate-200 rounded w-1/2" />
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── ColdStartBanner ────────────────────────────────────────────────────────
function ColdStartBanner({ onDismiss }) {
  return (
    <div className="mx-4 mt-3 flex items-start gap-2.5 bg-amber-50 border border-amber-200 rounded-xl px-3.5 py-3 fade-in">
      <svg className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd"/>
      </svg>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] font-bold text-amber-800 leading-snug">Backend waking up</p>
        <p className="text-[10px] text-amber-700 leading-relaxed mt-0.5">
          First request after inactivity may take 30–60 s while the server starts. Please wait — it&apos;s not broken!
        </p>
      </div>
      <button onClick={onDismiss} className="text-amber-400 hover:text-amber-600 transition-colors shrink-0">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>
  )
}

// ─── FileUploader ──────────────────────────────────────────────────────────
function FileUploader({ onUpload }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadingFile, setUploadingFile] = useState('')
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const uploadFiles = useCallback(async (files) => {
    if (!files || files.length === 0) return
    setError(null)
    setUploading(true)
    setUploadingFile(files.length === 1 ? files[0].name : `${files.length} files`)
    const form = new FormData()
    for (const file of files) form.append('files', file)
    try {
      const res = await fetch(`${API}/upload`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(120000),
      })
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Upload failed') }
      onUpload()
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
      setUploadingFile('')
    }
  }, [onUpload])

  const handleSampleDoc = useCallback(async (e) => {
    e.stopPropagation()
    try {
      const res = await fetch('/sample.txt')
      const blob = await res.blob()
      const file = new File([blob], 'AI-Introduction-Sample.txt', { type: 'text/plain' })
      await uploadFiles([file])
    } catch (e) {
      setError('Could not load sample document.')
    }
  }, [uploadFiles])

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false); uploadFiles(Array.from(e.dataTransfer.files))
  }

  return (
    <div className="flex flex-col gap-2 px-4 py-3 shrink-0">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 py-6 px-4
          ${dragging ? 'border-brand-500 bg-brand-50' : 'border-slate-300 hover:border-brand-400 bg-white hover:bg-slate-50'}`}
      >
        <input ref={inputRef} type="file" multiple accept=".pdf,.txt" className="hidden"
          onChange={(e) => uploadFiles(Array.from(e.target.files))} />
        {uploading ? (
          <div className="flex flex-col items-center gap-2.5 w-full text-center">
            <svg className="animate-spin w-5 h-5 text-brand-600 shrink-0" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            <div className="w-full flex flex-col gap-1 px-2">
              <span className="text-[11px] text-slate-500 font-semibold truncate max-w-full">Uploading {uploadingFile}…</span>
              <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-brand-600 rounded-full animate-[loading-bar_1.5s_infinite_ease-in-out]" style={{ width: '40%' }} />
              </div>
            </div>
          </div>
        ) : (
          <>
            <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
            </svg>
            <p className="text-xs text-slate-500 text-center font-medium">Drop PDF or .txt</p>
            <button className="mt-1 bg-brand-600 text-white text-[11px] font-semibold px-3 py-1.5 rounded-full shadow-sm hover:bg-brand-700 transition-colors">
              + Add document
            </button>
          </>
        )}
      </div>
      {/* Sample document shortcut */}
      {!uploading && (
        <button
          onClick={handleSampleDoc}
          className="flex items-center justify-center gap-1.5 text-[11px] text-brand-600 hover:text-brand-800 font-semibold py-1.5 rounded-lg hover:bg-brand-50 transition-colors border border-dashed border-brand-300 hover:border-brand-500"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"/>
          </svg>
          Try with a sample document
        </button>
      )}
      {error && (
        <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-1.5">{error}</p>
      )}
    </div>
  )
}

// ─── DocumentCard ───────────────────────────────────────────────────────────
function DocumentCard({ doc, onDelete }) {
  const [confirmDelete, setConfirmDelete] = useState(false)
  const isPdf = doc.doc_name.toLowerCase().endsWith('.pdf')

  if (confirmDelete) {
    return (
      <div className="flex flex-col gap-2 bg-red-50/50 border border-red-200 rounded-xl p-3 shadow-sm fade-in">
        <p className="text-[11px] text-red-700 font-semibold leading-normal">
          Remove this document?
        </p>
        <div className="flex items-center gap-1.5 self-end">
          <button
            onClick={() => setConfirmDelete(false)}
            className="text-[10px] text-slate-500 hover:text-slate-700 font-semibold px-2 py-1 rounded bg-white border border-slate-200 hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              setConfirmDelete(false)
              onDelete(doc.doc_name)
            }}
            className="text-[10px] text-white hover:bg-red-700 font-semibold px-2 py-1 rounded bg-red-600 transition-colors"
          >
            Confirm
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 bg-white border border-slate-200 rounded-xl px-3 py-3 group shadow-sm hover:shadow transition-shadow fade-in">
      {/* Icon */}
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${isPdf ? 'bg-amber-100 text-amber-600' : 'bg-blue-100 text-blue-600'}`}>
        {isPdf ? (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd"/>
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm3 1h6v4H7V5zm1 8a1 1 0 100 2h4a1 1 0 100-2H8z" clipRule="evenodd"/>
          </svg>
        )}
      </div>

      {/* Name + chunk count */}
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <p className="text-[13px] text-slate-700 font-semibold truncate leading-tight">
          {doc.doc_name}
        </p>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="bg-sky-100 text-sky-700 text-[10px] font-bold px-1.5 py-0.5 rounded">
            {doc.chunk_count} chunks
          </span>
        </div>
      </div>

      {/* Delete button */}
      <button
        onClick={() => setConfirmDelete(true)}
        title={`Remove ${doc.doc_name}`}
        className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-red-50 text-slate-400 hover:text-red-500 shrink-0"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
        </svg>
      </button>
    </div>
  )
}

// ─── DocumentManager ────────────────────────────────────────────────────────
function DocumentManager({ docs, fetching, onUpload, onDelete }) {
  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <FileUploader onUpload={onUpload} />

      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 pb-4 flex flex-col gap-2">
        {fetching ? (
          <LoadingSkeleton />
        ) : docs.length > 0 ? (
          docs.map((doc) => (
            <DocumentCard key={doc.doc_name} doc={doc} onDelete={onDelete} />
          ))
        ) : (
          <div className="text-center px-4 py-8 text-slate-400 text-xs">
            No documents uploaded yet.
          </div>
        )}
      </div>
    </div>
  )
}

// ─── ConfidenceBadge ─────────────────────────────────────────────────────────
function ConfidenceBadge({ confidence }) {
  const pct = Math.round(confidence * 100)
  const color = confidence >= 0.75
    ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
    : confidence >= 0.5
    ? 'bg-amber-100 text-amber-800 border-amber-200'
    : 'bg-red-100 text-red-800 border-red-200'
  return (
    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${color}`}>{pct}% Match</span>
  )
}

// ─── SourcePanel ─────────────────────────────────────────────────────────────
function SourcePanel({ sources, outOfScope }) {
  const [open, setOpen] = useState(false)

  if (outOfScope) {
    return (
      <div className="mt-2.5 flex items-center gap-2 text-xs text-slate-400 bg-slate-800/40 border border-slate-700/30 rounded-xl px-3.5 py-2.5 max-w-max">
        <svg className="w-4 h-4 text-slate-500 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
        </svg>
        <span>No relevant content found in your documents</span>
      </div>
    )
  }
  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-2 w-full">
      <button onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-brand-400 hover:text-brand-300 transition-colors font-semibold"
      >
        <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
          fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5"/>
        </svg>
        {open ? 'Hide' : 'Show'} {sources.length} source{sources.length > 1 ? 's' : ''}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2.5 fade-in w-full">
          {sources.map((src, i) => {
            const isPdf = src.doc_name.toLowerCase().endsWith('.pdf')
            return (
              <div key={i} className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-3.5 text-xs flex flex-col gap-2.5">
                <div className="flex items-center justify-between gap-2 border-b border-slate-800/60 pb-2">
                  <div className="flex items-center gap-2 min-w-0">
                    {isPdf ? (
                      <svg className="w-3.5 h-3.5 text-amber-500 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd"/>
                      </svg>
                    ) : (
                      <svg className="w-3.5 h-3.5 text-blue-500 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm3 1h6v4H7V5zm1 8a1 1 0 100 2h4a1 1 0 100-2H8z" clipRule="evenodd"/>
                      </svg>
                    )}
                    <span className="text-slate-300 font-medium truncate max-w-[140px]">{src.doc_name}</span>
                    <span className="text-slate-600">·</span>
                    <span className="text-slate-500 shrink-0">Page {src.page_num}</span>
                  </div>
                  <ConfidenceBadge confidence={src.confidence} />
                </div>
                <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">
                  <mark className="highlight-chunk">{src.text}</mark>
                </p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── TypingIndicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex items-end gap-2.5 fade-in">
      <div className="w-7 h-7 rounded-full bg-slate-800 flex items-center justify-center shrink-0">
        <svg className="w-4 h-4 text-slate-400" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd"/>
        </svg>
      </div>
      <div className="bg-slate-100 rounded-[16px] rounded-tl-[4px] px-4 py-3 flex gap-1.5 items-center">
        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-slate-400 inline-block animate-bounce" style={{ animationDelay: '0ms' }} />
        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-slate-400 inline-block animate-bounce" style={{ animationDelay: '150ms' }} />
        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-slate-400 inline-block animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  )
}

// ─── EmptyChatState ───────────────────────────────────────────────────────────
function EmptyChatState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-6 py-12 max-w-lg mx-auto w-full fade-in">
      <div className="w-24 h-24 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center mb-6 shadow-xl">
        <svg className="w-12 h-12 text-brand-600" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
        </svg>
      </div>
      <h3 className="text-base font-bold text-white mb-2">Upload a document to get started</h3>
      <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
        Upload PDF or TXT files to build your knowledge base. Llama 3.1 70B will answer your questions instantly based on exact segments of your documents.
      </p>
    </div>
  )
}

// ─── ChatThread ───────────────────────────────────────────────────────────────
function ChatThread({ messages, loading, hasDocuments }) {
  if (!hasDocuments) {
    return <EmptyChatState />
  }

  return (
    <div className="flex flex-col gap-6 px-6 py-6 max-w-4xl mx-auto w-full">
      {messages.length === 0 && !loading && (
        <div className="flex flex-col items-center justify-center gap-3 py-24 text-center text-slate-500 fade-in">
          <svg className="w-10 h-10 opacity-30 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"/>
          </svg>
          <p className="text-sm font-medium">Ask a question to begin chatting with your uploaded knowledge!</p>
        </div>
      )}
      {messages.map((msg, i) => {
        const renderFormattedContent = (text) => {
          if (!text) return ''
          const parts = text.split('**')
          return parts.map((part, index) => {
            if (index % 2 !== 0) {
              return (
                <strong key={index} className={`font-extrabold ${msg.role === 'user' ? 'text-white' : 'text-slate-950'}`}>
                  {part}
                </strong>
              )
            }
            return part
          })
        }
        return (
          <div key={i} className={`flex items-end gap-2.5 fade-in ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`} style={{ animationDuration: '200ms' }}>
            {msg.role === 'assistant' ? (
              <div className="w-7 h-7 rounded-full bg-slate-800 flex items-center justify-center shrink-0 mb-0.5 shadow-inner">
                <svg className="w-4 h-4 text-slate-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd"/>
                </svg>
              </div>
            ) : (
              <div className="w-7 h-7 shrink-0" />
            )}
            <div className={`flex flex-col max-w-[78%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`px-5 py-3 text-sm leading-relaxed whitespace-pre-wrap shadow-md transition-all duration-200
                ${msg.role === 'user'
                  ? 'bg-brand-600 text-white rounded-[16px] rounded-br-[4px]'
                  : 'bg-slate-100 text-slate-800 rounded-[16px] rounded-tl-[4px]'}`}>
                {renderFormattedContent(msg.content)}
              </div>
              {msg.role === 'assistant' && (
                <SourcePanel sources={msg.sources} outOfScope={msg.out_of_scope} />
              )}
            </div>
          </div>
        )
      })}
      {loading && <TypingIndicator />}
    </div>
  )
}

// ─── MessageInput ─────────────────────────────────────────────────────────────
function MessageInput({ onSend, loading, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || loading || disabled) return
    onSend(trimmed); setText('')
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="flex items-end gap-2 p-4 max-w-4xl mx-auto w-full">
      <div className="flex-1 flex items-end bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden focus-within:border-brand-500 transition-colors">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? 'Upload a document first…' : 'Ask a question…'}
          rows={1}
          disabled={loading || disabled}
          className="flex-1 bg-transparent resize-none px-4 py-3.5 text-sm text-slate-200 placeholder-slate-500 outline-none disabled:opacity-50 max-h-36 scrollbar-thin rounded-2xl"
          style={{ overflowY: 'auto' }}
          onInput={(e) => {
            e.target.style.height = 'auto'
            e.target.style.height = Math.min(e.target.scrollHeight, 144) + 'px'
          }}
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || loading || disabled}
          className="m-2 p-2 rounded-full bg-brand-600 text-white hover:bg-brand-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all active:scale-95 flex items-center justify-center shrink-0 w-9.5 h-9.5"
        >
          {loading ? (
            <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : (
            <svg className="w-4 h-4 text-white transform rotate-45 -translate-x-[1px] translate-y-[1px]" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"/>
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}

// ─── SplashScreen ───────────────────────────────────────────────────────────
function SplashScreen({ fading }) {
  return (
    <div
      className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-slate-950"
      style={{ transition: 'opacity 0.5s ease', opacity: fading ? 0 : 1, pointerEvents: fading ? 'none' : 'all' }}
    >
      <div className="flex flex-col items-center gap-5">
        {/* Logo mark */}
        <div className="w-20 h-20 rounded-2xl bg-[#0F6E56] flex items-center justify-center shadow-2xl" style={{ boxShadow: '0 0 60px rgba(15,110,86,0.45)' }}>
          <span className="text-white font-bold" style={{ fontSize: 44, lineHeight: 1, fontFamily: 'Inter, system-ui, sans-serif', letterSpacing: '-2px' }}>Q</span>
        </div>
        {/* Brand name */}
        <div className="flex flex-col items-center gap-1">
          <h1 className="text-white font-bold tracking-tight" style={{ fontSize: 32, letterSpacing: '-1px' }}>Queryq</h1>
          <p className="text-slate-400 text-sm font-medium">AI-Powered Document Chat</p>
        </div>
        {/* Animated loading dots */}
        <div className="flex gap-1.5 mt-2">
          {[0, 1, 2].map(i => (
            <div key={i} className="w-1.5 h-1.5 rounded-full bg-[#0F6E56]"
              style={{ animation: 'splash-dot 1.2s ease-in-out infinite', animationDelay: `${i * 0.2}s` }}
            />
          ))}
        </div>
      </div>
      <style>{`
        @keyframes splash-dot {
          0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
    </div>
  )
}

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [docs, setDocs]         = useState([])   // { doc_name, chunk_count }
  const [fetching, setFetching] = useState(true)
  const [messages, setMessages] = useState([])
  const [loading, setLoading]   = useState(false)
  const [deleteError, setDeleteError] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [showColdStart, setShowColdStart] = useState(true)
  const [appReady, setAppReady]   = useState(false)   // false = splash visible
  const [splashFading, setSplashFading] = useState(false) // triggers CSS fade-out
  const threadRef = useRef(null)

  // Show splash for 1.4s, then fade it out over 0.5s
  useEffect(() => {
    const fadeTimer = setTimeout(() => setSplashFading(true), 1400)
    const hideTimer = setTimeout(() => setAppReady(true), 1900)
    return () => { clearTimeout(fadeTimer); clearTimeout(hideTimer) }
  }, [])

  const fetchDocuments = useCallback(async () => {
    setFetching(true)
    try {
      const res = await fetch(`${API}/documents`)
      if (res.ok) {
        const data = await res.json()
        setDocs(data.documents || [])
      }
    } catch (e) {
      console.error('Failed to fetch documents:', e)
    } finally {
      setFetching(false)
    }
  }, [])

  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

  const scrollToBottom = () => {
    setTimeout(() => {
      threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'smooth' })
    }, 50)
  }

  const handleDelete = async (docName) => {
    setDeleteError(null)
    try {
      const res = await fetch(`${API}/documents/${encodeURIComponent(docName)}`, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Delete failed')
      }
      await fetchDocuments()
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `📄 "${docName}" has been removed from the session.`,
        sources: [], out_of_scope: false,
      }])
    } catch (e) {
      setDeleteError(e.message)
    }
  }

  const handleSend = async (question) => {
    const userMsg = { role: 'user', content: question }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    scrollToBottom()
    setLoading(true)

    const historyForApi = newMessages.slice(0, -1).map(m => ({ role: m.role, content: m.content }))

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history: historyForApi }),
        signal: AbortSignal.timeout(120000), // 2 minute timeout
      })
      
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Chat request failed')
      }
      
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        out_of_scope: data.out_of_scope,
      }])
    } catch (e) {
      // Detect Groq rate-limit (429) and show a friendly message
      const isRateLimit = e.message?.includes('429') || e.message?.includes('rate_limit') || e.message?.includes('Rate limit')
      const errorMsg = isRateLimit
        ? '⏳ Rate limit reached — the free Groq tier has hit its daily token limit. Please wait a few minutes and try again, or use a different API key.'
        : `⚠️ Error: ${e.message}`
        
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorMsg,
        sources: [], out_of_scope: false,
      }])
    } finally {
      setLoading(false)
      scrollToBottom()
    }
  }

  const hasDocuments = docs.length > 0

  return (
    <>
      {/* Branded splash — stays until appReady, fades out smoothly */}
      {!appReady && <SplashScreen fading={splashFading} />}

      <div className="flex h-screen overflow-hidden bg-white relative">

      {/* ── Left Sidebar Backdrop (Mobile overlay) ────────────────────────── */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-40 md:hidden transition-all duration-200"
        />
      )}

      {/* ── Left Sidebar (260px) ──────────────────────────────────────────────── */}
      <aside className={`w-[260px] shrink-0 flex flex-col bg-slate-50 border-r border-slate-200
        fixed inset-y-0 left-0 z-50 md:relative md:translate-x-0 transition-transform duration-200 ease-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}
      >
        
        {/* App Logo */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 bg-white">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center shrink-0 shadow-sm">
              <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd"/>
              </svg>
            </div>
            <p className="text-[14px] font-bold text-slate-800 tracking-tight">Querq</p>
          </div>
          {/* Close button inside sidebar on mobile */}
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-600 md:hidden transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Cold-start banner */}
        {showColdStart && <ColdStartBanner onDismiss={() => setShowColdStart(false)} />}

        {/* Document Manager */}
        <DocumentManager docs={docs} fetching={fetching} onUpload={fetchDocuments} onDelete={handleDelete} />
        {deleteError && (
          <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg mx-4 mb-3 px-3 py-1.5">{deleteError}</p>
        )}
      </aside>

      {/* ── Main chat area ───────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 bg-slate-950 text-slate-100 relative">
        {/* Header */}
        <header className="flex items-center gap-3 px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm shrink-0">
          {/* Drawer trigger button */}
          <button
            onClick={() => setSidebarOpen(o => !o)}
            className="p-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-slate-300 md:hidden transition-colors shrink-0"
            title="Toggle Sidebar"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
            </svg>
          </button>

          <div className="flex-1 min-w-0">
            <h1 className="text-[14px] font-bold text-white tracking-tight truncate">Ask anything about your documents</h1>
          </div>
          
          <div className="flex items-center gap-4 shrink-0">
            {/* Llama purple badge */}
            <span className="inline-flex items-center gap-1 bg-purple-950/40 border border-purple-800/60 text-purple-300 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider shrink-0">
              <svg className="w-3 h-3 text-purple-400 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd"/>
              </svg>
              Llama 3.1 70B
            </span>

            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="text-xs text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1 shrink-0"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"/>
                </svg>
                Clear
              </button>
            )}
          </div>
        </header>

        {/* Thread */}
        <div ref={threadRef} className="flex-1 overflow-y-auto scrollbar-thin">
          <ChatThread messages={messages} loading={loading} hasDocuments={hasDocuments} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-800 bg-slate-900/50 shrink-0">
          <MessageInput onSend={handleSend} loading={loading} disabled={!hasDocuments} />
        </div>
      </main>
    </div>
    </>
  )
}
