<script setup lang="ts">
import { ref } from 'vue'
import { uploadDocument, type IngestResult } from '../api/ingest'

interface DocItem {
  name: string
  status: 'uploading' | 'success' | 'error'
  result?: IngestResult
  error?: string
}

const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const docs = ref<DocItem[]>([])

const SUPPORTED_EXTS = ['.pdf', '.png', '.jpg', '.jpeg', '.csv', '.xlsx', '.xls', '.txt', '.md']

function pickFiles() {
  fileInput.value?.click()
}

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files) uploadFiles(Array.from(target.files))
  target.value = ''
}

function onDrop(e: DragEvent) {
  isDragging.value = false
  if (e.dataTransfer?.files) uploadFiles(Array.from(e.dataTransfer.files))
}

function onDragOver() {
  isDragging.value = true
}

function onDragLeave() {
  isDragging.value = false
}

async function uploadFiles(files: File[]) {
  for (const file of files) {
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    if (!SUPPORTED_EXTS.includes(ext)) {
      docs.value.unshift({
        name: file.name,
        status: 'error',
        error: `不支持的文件类型: ${ext}`,
      })
      continue
    }
    const item: DocItem = { name: file.name, status: 'uploading' }
    docs.value.unshift(item)
    try {
      const result = await uploadDocument(file)
      item.status = 'success'
      item.result = result
    } catch (e) {
      item.status = 'error'
      item.error = (e as Error).message
    }
  }
}
</script>

<template>
  <div>
    <div class="mb-6">
      <h2 class="text-lg font-medium text-gray-900">文档入库</h2>
      <p class="text-sm text-gray-500 mt-0.5">
        上传文档自动解析、向量化并构建知识图谱
      </p>
    </div>

    <div
      @click="pickFiles"
      @drop="onDrop"
      @dragover.prevent="onDragOver"
      @dragleave="onDragLeave"
      class="border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors"
      :class="isDragging ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-gray-400'"
    >
      <svg class="mx-auto text-gray-400" width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <p class="mt-3 text-sm text-gray-700">点击或拖拽文件到此处上传</p>
      <p class="mt-1 text-xs text-gray-500">支持 PDF · PNG/JPG · CSV/XLSX · TXT · MD</p>
      <input
        ref="fileInput"
        type="file"
        multiple
        accept=".pdf,.png,.jpg,.jpeg,.csv,.xlsx,.xls,.txt,.md"
        class="hidden"
        @change="onFileChange"
      />
    </div>

    <div v-if="docs.length" class="mt-6 space-y-2">
      <div
        v-for="(doc, i) in docs"
        :key="i"
        class="bg-white border border-gray-200 rounded-lg px-4 py-3 flex items-center gap-3"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-emerald-500 shrink-0">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-gray-900 truncate">{{ doc.name }}</p>
          <p v-if="doc.status === 'uploading'" class="text-xs text-indigo-600 mt-0.5">处理中...</p>
          <p v-else-if="doc.status === 'success' && doc.result" class="text-xs text-gray-500 mt-0.5">
            {{ doc.result.chunks_count }} 块 · {{ doc.result.entities_count }} 实体 · {{ doc.result.relations_count }} 关系
          </p>
          <p v-else-if="doc.status === 'error'" class="text-xs text-red-600 mt-0.5">{{ doc.error }}</p>
        </div>
        <span
          v-if="doc.status !== 'uploading'"
          class="w-2 h-2 rounded-full shrink-0"
          :class="doc.status === 'success' ? 'bg-emerald-500' : 'bg-red-500'"
        />
        <span v-else class="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin shrink-0" />
      </div>
    </div>
  </div>
</template>
