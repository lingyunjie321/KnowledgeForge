<script setup lang="ts">
import { ref, nextTick } from 'vue'
import { askStream, type QAMeta, type QASource, type RetrieveStep } from '../api/qa'

interface ChatMessage {
  role: 'user' | 'agent'
  content: string
  meta?: QAMeta
  error?: boolean
  streaming?: boolean
}

const input = ref('')
const asking = ref(false)
const messages = ref<ChatMessage[]>([
  {
    role: 'agent',
    content: '你好，上传文档后可以基于知识图谱和 GraphRAG 进行混合检索问答。输入问题开始吧。',
  },
])
const containerRef = ref<HTMLElement | null>(null)

async function scrollToBottom() {
  await nextTick()
  if (containerRef.value) {
    containerRef.value.scrollTop = containerRef.value.scrollHeight
  }
}

async function send() {
  const question = input.value.trim()
  if (!question || asking.value) return

  asking.value = true
  input.value = ''
  messages.value.push({ role: 'user', content: question })

  const agentMsg: ChatMessage = { role: 'agent', content: '', streaming: true }
  messages.value.push(agentMsg)

  try {
    for await (const evt of askStream(question)) {
      if (evt.type === 'meta') {
        agentMsg.meta = {
          intent: evt.intent,
          confidence: evt.confidence,
          reasoning_steps: evt.reasoning_steps,
          sources: evt.sources,
          retrieve_steps: evt.retrieve_steps,
        }
      } else if (evt.type === 'token') {
        agentMsg.content += evt.content
      } else if (evt.type === 'done') {
        if (agentMsg.meta) {
          agentMsg.meta.reasoning_steps = evt.reasoning_steps
        }
      } else if (evt.type === 'error') {
        agentMsg.error = true
        agentMsg.content = `请求失败：${evt.message}`
      }
      await scrollToBottom()
    }
  } catch (e) {
    agentMsg.error = true
    agentMsg.content = `请求失败：${(e as Error).message}`
  } finally {
    agentMsg.streaming = false
    asking.value = false
  }
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

const STEP_COLORS: Record<string, string> = {
  vector: 'bg-blue-500',
  entity_linking: 'bg-purple-500',
  subgraph: 'bg-emerald-500',
  path: 'bg-amber-500',
  community: 'bg-pink-500',
  rerank: 'bg-gray-700',
}

function totalCost(steps: RetrieveStep[] | undefined): number {
  if (!steps?.length) return 0
  return steps.reduce((sum, s) => sum + s.cost_ms, 0)
}

function metadataText(source: QASource): string {
  const metadata = source.metadata ?? {}
  const parts = ['doc_id', 'chunk_id', 'entity', 'from', 'to']
    .map((key) => {
      const value = metadata[key]
      return typeof value === 'string' && value ? `${key}: ${value}` : ''
    })
    .filter(Boolean)
  return parts.join(' · ')
}
</script>

<template>
  <div class="flex flex-col h-[calc(100vh-140px)]">
    <div ref="containerRef" class="flex-1 overflow-y-auto space-y-4 pr-2">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        class="flex gap-3"
        :class="msg.role === 'user' ? 'flex-row-reverse' : ''"
      >
        <div
          class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium shrink-0"
          :class="msg.role === 'user' ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-600'"
        >
          {{ msg.role === 'user' ? '我' : 'AI' }}
        </div>
        <div class="max-w-[80%] space-y-2">
          <div
            class="rounded-2xl px-4 py-2.5 text-sm leading-relaxed"
            :class="
              msg.role === 'user'
                ? 'bg-indigo-600 text-white'
                : msg.error
                  ? 'bg-red-50 text-red-700 border border-red-200'
                  : 'bg-white border border-gray-200 text-gray-800'
            "
          >
            <p v-if="!msg.content && msg.streaming" class="flex gap-1 py-1">
              <span class="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style="animation-delay: 0ms" />
              <span class="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style="animation-delay: 150ms" />
              <span class="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style="animation-delay: 300ms" />
            </p>
            <template v-else>
              <p v-for="(line, j) in msg.content.split('\n')" :key="j" class="min-h-[1em]">
                {{ line || '\u00A0' }}
              </p>
            </template>
          </div>

          <div v-if="msg.meta" class="flex flex-wrap gap-2 text-xs">
            <span class="px-2 py-0.5 rounded bg-purple-50 text-purple-700 border border-purple-200">
              {{ msg.meta.intent }}
            </span>
            <span class="px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
              置信度 {{ Math.round(msg.meta.confidence * 100) }}%
            </span>
          </div>

          <details
            v-if="msg.meta && msg.meta.retrieve_steps && msg.meta.retrieve_steps.length"
            class="text-xs"
            open
          >
            <summary class="cursor-pointer text-gray-600 hover:text-gray-800 select-none font-medium">
              GraphRAG 6 步检索 · 总耗时 {{ totalCost(msg.meta.retrieve_steps) }} ms
            </summary>
            <ol class="mt-2 space-y-1.5">
              <li
                v-for="(step, k) in msg.meta.retrieve_steps"
                :key="k"
                class="flex items-start gap-2 pl-1"
              >
                <span
                  class="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
                  :class="STEP_COLORS[step.name] ?? 'bg-gray-400'"
                />
                <div class="flex-1">
                  <div class="flex items-baseline gap-2">
                    <span class="text-gray-700 font-medium">{{ step.label }}</span>
                    <span class="text-gray-400">{{ step.cost_ms }} ms</span>
                    <span class="text-gray-500">命中 {{ step.hits }}</span>
                  </div>
                  <p v-if="step.detail" class="text-gray-500 mt-0.5">{{ step.detail }}</p>
                </div>
              </li>
            </ol>
          </details>

          <details v-else-if="msg.meta && msg.meta.reasoning_steps.length" class="text-xs">
            <summary class="cursor-pointer text-gray-500 hover:text-gray-700 select-none">
              推理过程 ({{ msg.meta.reasoning_steps.length }} 步)
            </summary>
            <ol class="mt-2 space-y-1 pl-4 list-decimal text-gray-600">
              <li v-for="(step, k) in msg.meta.reasoning_steps" :key="k" class="leading-relaxed">
                {{ step }}
              </li>
            </ol>
          </details>

          <details v-if="msg.meta && msg.meta.sources.length" class="text-xs">
            <summary class="cursor-pointer text-gray-500 hover:text-gray-700 select-none">
              参考来源 ({{ msg.meta.sources.length }})
            </summary>
            <ul class="mt-2 space-y-1.5 pl-4 text-gray-600">
              <li v-for="(s, k) in msg.meta.sources" :key="k" class="leading-relaxed">
                <span class="text-gray-400">[{{ k + 1 }}]</span>
                <span class="ml-1 px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">{{ s.type }}</span>
                <span class="ml-1 text-gray-400">{{ Math.round(s.score * 100) }}%</span>
                <span class="ml-1 text-gray-700">{{ s.source }}</span>
                <span v-if="metadataText(s)" class="ml-1 text-gray-400">{{ metadataText(s) }}</span>
              </li>
            </ul>
          </details>
        </div>
      </div>
    </div>

    <div class="mt-4 flex gap-2">
      <input
        v-model="input"
        @keydown="onKeydown"
        :disabled="asking"
        placeholder="输入问题，Enter 发送"
        class="flex-1 px-4 py-2.5 rounded-lg border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-50"
      />
      <button
        @click="send"
        :disabled="asking || !input.trim()"
        class="px-5 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
      >
        {{ asking ? '生成中' : '发送' }}
      </button>
    </div>
  </div>
</template>
